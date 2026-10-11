from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from sentinal.compute.backend import ComputeBackend

logger = logging.getLogger(__name__)

try:
    import Metal
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


@dataclass(frozen=True)
class MetalKernel:
    """Descriptor for a Metal compute kernel dispatch."""
    shader_source: str
    function_name: str
    input_data: list  # list of bytes-like objects
    output_size: int
    grid_size: tuple[int, int, int]
    group_size: tuple[int, int, int]


class MetalBackend(ComputeBackend):
    """Dispatches raw Metal compute shaders to Apple GPU."""

    def __init__(self):
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal framework not available")

        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("Failed to create Metal device")

        self._pipeline_cache: dict[tuple[str, str], Any] = {}
        self._command_queue = self._device.newCommandQueue()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def name(self) -> str:
        return "metal"

    def is_available(self) -> bool:
        """Check if Metal device is available."""
        if not METAL_AVAILABLE:
            return False
        device = Metal.MTLCreateSystemDefaultDevice()
        return device is not None

    def device_info(self) -> dict:
        """Returns device information."""
        return {
            "name": self._device.name(),
            "unified_memory": self._device.hasUnifiedMemory()
        }

    async def dispatch(self, kernel_or_fn, *args, **kwargs) -> Any:
        """Dispatch a kernel or callable function.

        If kernel_or_fn is a MetalKernel, dispatches it to the GPU.
        Otherwise, treats it as a callable and runs it via executor.
        """
        if isinstance(kernel_or_fn, MetalKernel):
            return await self._dispatch_kernel(kernel_or_fn)
        else:
            return await self.dispatch_fn(kernel_or_fn, *args, **kwargs)

    async def dispatch_fn(self, fn: Callable, *args, **kwargs) -> Any:
        """Run a callable via executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: fn(*args, **kwargs)
        )

    def profile(self, kernel_or_fn, *args, **kwargs) -> float:
        """Profile execution time of a kernel or callable."""
        start = time.perf_counter()

        if isinstance(kernel_or_fn, MetalKernel):
            # Run synchronously for profiling
            asyncio.run(self._dispatch_kernel(kernel_or_fn))
        else:
            kernel_or_fn(*args, **kwargs)

        return time.perf_counter() - start

    async def _dispatch_kernel(self, kernel: MetalKernel) -> memoryview:
        """Core Metal dispatch logic.

        Compiles the shader, creates buffers, dispatches the compute kernel,
        and returns the output as a memoryview.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._dispatch_kernel_sync, kernel
        )

    def _dispatch_kernel_sync(self, kernel: MetalKernel) -> memoryview:
        """Synchronous Metal kernel dispatch."""
        logger.debug(f"Dispatching Metal kernel: {kernel.function_name}")
        # Get or compile pipeline
        pipeline = self._get_pipeline(kernel.shader_source, kernel.function_name)

        # Create input buffers
        input_buffers = []
        for data in kernel.input_data:
            if not isinstance(data, bytes):
                data = bytes(data)
            buf = self._device.newBufferWithBytes_length_options_(
                data,
                len(data),
                Metal.MTLResourceStorageModeShared
            )
            input_buffers.append(buf)

        # Create output buffer
        out_buf = self._device.newBufferWithLength_options_(
            kernel.output_size,
            Metal.MTLResourceStorageModeShared
        )

        # Create command buffer and encoder
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        # Set pipeline and buffers
        encoder.setComputePipelineState_(pipeline)
        for i, buf in enumerate(input_buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        encoder.setBuffer_offset_atIndex_(out_buf, 0, len(input_buffers))

        # Dispatch threads
        grid = Metal.MTLSizeMake(*kernel.grid_size)
        group = Metal.MTLSizeMake(*kernel.group_size)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid, group)

        # Commit and wait
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Return output as memoryview
        return memoryview(out_buf.contents().as_buffer(kernel.output_size))

    def _get_pipeline(self, shader_source: str, function_name: str):
        """Compile shader and cache pipeline."""
        cache_key = (shader_source, function_name)

        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        # Compile shader
        library = self._device.newLibraryWithSource_options_error_(
            shader_source, None, None
        )
        if library is None or library[0] is None:
            raise RuntimeError(f"Failed to compile Metal shader: {library[2] if library else 'unknown error'}")

        library = library[0]
        function = library.newFunctionWithName_(function_name)
        if function is None:
            raise RuntimeError(f"Function '{function_name}' not found in shader")

        # Create pipeline state
        pipeline = self._device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None or pipeline[0] is None:
            raise RuntimeError(f"Failed to create pipeline state: {pipeline[2] if pipeline else 'unknown error'}")

        pipeline = pipeline[0]
        self._pipeline_cache[cache_key] = pipeline
        return pipeline
