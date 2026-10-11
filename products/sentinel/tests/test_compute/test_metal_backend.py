import asyncio
import struct
import pytest

from sentinal.compute.metal_backend import MetalBackend, MetalKernel

try:
    import Metal
    METAL_AVAILABLE = Metal.MTLCreateSystemDefaultDevice() is not None
except ImportError:
    METAL_AVAILABLE = False

# Skip all tests if Metal is not available
pytestmark = pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")


class TestMetalBackend:
    """Test MetalBackend compute operations."""

    def test_name(self):
        """Test that backend name is 'metal'."""
        backend = MetalBackend()
        assert backend.name == "metal"

    def test_is_available(self):
        """Test Metal availability check."""
        backend = MetalBackend()
        assert backend.is_available() is True

    def test_device_info(self):
        """Test device info contains Apple GPU information."""
        backend = MetalBackend()
        info = backend.device_info()
        assert "name" in info
        assert "Apple" in info["name"]
        assert "unified_memory" in info

    def test_dispatch_kernel_vector_add(self):
        """Test dispatching a vector_add kernel on 1024 floats."""
        backend = MetalBackend()

        # Vector add shader
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void vector_add(
            device const float* a [[buffer(0)]],
            device const float* b [[buffer(1)]],
            device float* out [[buffer(2)]],
            uint id [[thread_position_in_grid]]
        ) {
            out[id] = a[id] + b[id];
        }
        """

        # Create test data
        n = 1024
        a_values = [float(i) for i in range(n)]
        b_values = [float(i * 2) for i in range(n)]

        # Pack into bytes
        a_bytes = struct.pack(f'{n}f', *a_values)
        b_bytes = struct.pack(f'{n}f', *b_values)

        # Create kernel
        kernel = MetalKernel(
            shader_source=shader_source,
            function_name="vector_add",
            input_data=[a_bytes, b_bytes],
            output_size=n * 4,  # 4 bytes per float
            grid_size=(n, 1, 1),
            group_size=(256, 1, 1)
        )

        # Dispatch kernel
        result_memoryview = asyncio.run(backend.dispatch(kernel))

        # Unpack and verify output
        result = struct.unpack(f'{n}f', bytes(result_memoryview))

        # Verify correctness
        for i in range(n):
            expected = a_values[i] + b_values[i]
            assert abs(result[i] - expected) < 1e-5, f"Mismatch at index {i}: {result[i]} != {expected}"

    def test_dispatch_fn(self):
        """Test dispatching a regular callable function."""
        backend = MetalBackend()

        def test_fn(x, y):
            return x + y

        result = asyncio.run(backend.dispatch(test_fn, 5, 3))
        assert result == 8

    def test_profile_kernel(self):
        """Test profiling a simple kernel."""
        backend = MetalBackend()

        # Simple shader
        shader_source = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void simple_copy(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            uint id [[thread_position_in_grid]]
        ) {
            output[id] = input[id];
        }
        """

        # Create small test data
        n = 256
        input_values = [float(i) for i in range(n)]
        input_bytes = struct.pack(f'{n}f', *input_values)

        # Create kernel
        kernel = MetalKernel(
            shader_source=shader_source,
            function_name="simple_copy",
            input_data=[input_bytes],
            output_size=n * 4,
            grid_size=(n, 1, 1),
            group_size=(64, 1, 1)
        )

        # Profile the kernel
        elapsed = backend.profile(kernel)

        # Verify elapsed time is reasonable
        assert elapsed > 0, "Elapsed time should be positive"
        assert elapsed < 1.0, f"Elapsed time too large: {elapsed}s"
