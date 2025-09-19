from concurrent.futures import Future
from typing import Sequence, Union

import numpy as np

from ..core import Detector


class HDRCamera(Detector):
    """Wrapper to convert a camera to a high dynamic range (HDR) camera.

    This wrapper forwards all method calls and attribute acces to the camera object,
    except for the trigger method, which is replaced by a method that triggers the camera
    multiple times with different exposure times.

    todo: find out why the HDR camera is so slow.

    Args:
        camera: The camera object to wrap.
        background: The background value to subtract from each image before scaling.
            If a sequence is passed, each value corresponds to the background for a different exposure factor.
        saturation_threshold: The threshold value to consider a pixel saturated, in which case it will not be taken into account in the computation of the result.
        exposure_factors: A sequence of exposure factors to use for HDR imaging.
    """

    def __init__(
        self,
        camera,
        background: Union[Sequence[float], float],
        saturation_threshold: int,
        exposure_factors: Sequence[float] = (1.0, 0.1, 0.01),
    ):
        self._camera = camera  # object.__setattr__(self, "_camera", camera)
        self._exposure_factors = np.asarray(exposure_factors, dtype=np.float32)
        self._saturation_threshold = saturation_threshold
        self._background = (background,) * len(exposure_factors) if np.isscalar(background) else background
        super().__init__(
            data_shape=camera.data_shape,
            pixel_size=camera.pixel_size,
            duration=camera.duration,
            latency=camera.latency,
            multi_threaded=False,
        )

    def trigger(self, *args, out=None, immediate=False, **kwargs) -> Future:
        """Trigger the camera multiple times with different exposure times.

        todo: at the moment, this method is always blocking, the immediate flag is ignored.

        Args:
            *args: Positional arguments to pass to the camera's trigger method.
            out: Optional output buffer.
            immediate: If True, trigger the camera immediately.
            **kwargs: Keyword arguments to pass to the camera's trigger method.

        Returns:
            A Future object representing the result of the trigger operation.
        """
        exposure = self._camera.exposure
        data = None
        weight = None
        for factor, background in zip(self._exposure_factors, self._background):
            self._camera.exposure = exposure * factor
            frame = (self._camera.trigger(*args, out=out, immediate=True, **kwargs).result()).astype(
                np.float32
            ) - background
            mask = frame < self._saturation_threshold - background
            frame[~mask] = 0
            if data is None:
                data = frame
                weight = mask * factor
            else:
                data += frame
                weight += mask * factor

        # finally divide by zero, special treatment if there were zeros in the mask
        mask = weight == 0
        weight[mask] = 1.0
        data[mask] = self._saturation_threshold
        data /= weight
        if out is not None:
            out[...] = data

        # Set the exposure back to the original value
        self._camera.exposure = exposure
        result = Future()
        result.set_result(data)
        return result

    def _fetch(self, *args, **kwargs) -> np.ndarray:
        """Fetch method required by the Detector interface.

        This method is not implemented for HDRCamera because the trigger method
        directly returns the result without using the fetch mechanism.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: Always raised because this method is not implemented.
        """
        raise NotImplementedError("Fetch is not implemented for HDRCamera")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped camera object.

        This method is called when an attribute is not found in the HDRCamera instance.
        It forwards the attribute access to the wrapped camera object, except for
        attributes starting with an underscore.

        Args:
            name: The name of the attribute to access.

        Returns:
            The attribute value from the wrapped camera object.
        """
        # Forward attribute access to the target
        if name.startswith("_"):
            return super().__getattr__(name)
        else:
            return getattr(self._camera, name)

    def __setattr__(self, name, value):
        """Forward attribute setting to the wrapped camera object.

        This method is called when an attribute is set on the HDRCamera instance.
        It forwards the attribute setting to the wrapped camera object, except for
        attributes starting with an underscore.

        Args:
            name: The name of the attribute to set.
            value: The value to set the attribute to.
        """
        # Forward attribute setting to the target
        if name.startswith("_"):
            return super().__setattr__(name, value)
        else:
            setattr(self._camera, name, value)
