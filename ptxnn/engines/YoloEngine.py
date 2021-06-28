from .GeneralEngine import GeneralInferenceEngine
from .constants import CHECK_TIMEOUT
from _ptxnn import YoloEngine as YoloEngine_impl

import typing as tp
import numpy as np
import asyncio

class YoloEngine(GeneralInferenceEngine):

    def __init__(self, model_name: str, engine_path: str, n_classes: int, conf_threshold: float) -> None:
        """Constructs a new 'GeneralInferenceEngine' object
            
        Args:
          model_name: used for log message
          engine_path: path to engine file
          n_classes: count of output classes, used for decoding
          conf_threshold: probability threshold for detections
        Raises:
          RuntimeError: If file not found
        """
        self.engine = YoloEngine_impl(model_name, engine_path, n_classes, conf_threshold)

    def predict_images(self, images: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
        """Perform model predict on list of images.
        Args:
          images: List of images with different sizes, but all encoded in BGR uint8 format and have value in [0;255]
        Returns:
          List of detections in relative format [x0;y0;x1;y1;class_id;prob]
        Raises:
          RuntimeError: if some CUDA exception happens
        """
        return self.engine.predict_image(images)

class AsyncYoloEngine(YoloEngine):
    """Asynchronous version of YoloEngine
    """

    def __init__(self, model_name: str, engine_path: str, n_classes: int, conf_threshold: float) -> None:
        super().__init__(model_name, engine_path, n_classes, conf_threshold)
        self.enqueue_condition = asyncio.Condition()
        self.predicted_condition = asyncio.Condition()
        asyncio.ensure_future(self._notifier(), loop=asyncio.get_event_loop())

    async def _notifier(self):
        """Internal method to notify Future object, that execution is ended
        """
        while asyncio.get_running_loop().is_running():
            # First block execution until tasks not enqueued 
            async with self.enqueue_condition:
                await self.enqueue_condition.wait()
            
            # Than polling engine to check execution progress
            while True:
                inferencing = self.engine.is_inferencing()
                if not inferencing:
                    async with self.predicted_condition:
                        self.predicted_condition.notify_all()
                    break
                else:
                    await asyncio.sleep(CHECK_TIMEOUT)

    def predict_images(self, input_data: np.ndarray) -> tp.List[np.ndarray]:
        async def coro():
            self.engine.predict_image_callbacked(input_data)

            # send information to _notifier task, that execution started
            async with self.enqueue_condition:
                self.enqueue_condition.notify_all()

            # than wait until prediction is done
            async with self.predicted_condition:
                await self.predicted_condition.wait_for(lambda: not self.engine.is_inferencing())
            
            return self.engine.synchronize_image_callback()
        
        return coro()