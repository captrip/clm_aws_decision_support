import asyncio
import pytest
from typing import Any, Callable

from sentinal.compute.device import DevicePreference
from sentinal.compute.backend import ComputeBackend


class TestDevicePreference:
    """Test DevicePreference enum values."""

    def test_cpu_value(self):
        """Test CPU enum value is 'cpu'."""
        assert DevicePreference.CPU.value == "cpu"

    def test_gpu_value(self):
        """Test GPU enum value is 'gpu'."""
        assert DevicePreference.GPU.value == "gpu"

    def test_auto_value(self):
        """Test AUTO enum value is 'auto'."""
        assert DevicePreference.AUTO.value == "auto"

    def test_all_members(self):
        """Test all enum members are present."""
        members = {member.name for member in DevicePreference}
        assert members == {"CPU", "GPU", "AUTO"}


class TestComputeBackendABC:
    """Test ComputeBackend abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that ComputeBackend ABC cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ComputeBackend()

    def test_incomplete_subclass_raises_error(self):
        """Test that incomplete subclass missing abstract methods raises TypeError."""

        class IncompleteBackend(ComputeBackend):
            """Subclass missing all required methods."""
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteBackend()

    def test_partial_subclass_raises_error(self):
        """Test that partial implementation still raises TypeError."""

        class PartialBackend(ComputeBackend):
            """Subclass with only some methods implemented."""

            @property
            def name(self) -> str:
                return "partial"

            def is_available(self) -> bool:
                return True
            # Missing dispatch and profile

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PartialBackend()

    def test_valid_subclass_can_be_instantiated(self):
        """Test that a complete subclass with all methods can be instantiated."""

        class ValidBackend(ComputeBackend):
            """Complete implementation of ComputeBackend."""

            @property
            def name(self) -> str:
                return "valid"

            def is_available(self) -> bool:
                return True

            async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
                return fn(*args, **kwargs)

            def profile(self, fn: Callable, *args, **kwargs) -> float:
                return 0.0

        # Should not raise any exception
        backend = ValidBackend()
        assert backend.name == "valid"
        assert backend.is_available() is True
        assert backend.profile(lambda: None) == 0.0

    def test_valid_subclass_dispatch_method(self):
        """Test that dispatch method works correctly in a valid subclass."""

        class ValidBackend(ComputeBackend):
            """Complete implementation of ComputeBackend."""

            @property
            def name(self) -> str:
                return "valid"

            def is_available(self) -> bool:
                return True

            async def dispatch(self, fn: Callable, *args, **kwargs) -> Any:
                return fn(*args, **kwargs)

            def profile(self, fn: Callable, *args, **kwargs) -> float:
                return 0.0

        backend = ValidBackend()
        result = asyncio.run(backend.dispatch(lambda x: x * 2, 5))
        assert result == 10
