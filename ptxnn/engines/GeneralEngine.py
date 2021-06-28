from _ptxnn import GeneralInferenceEngine as GeneralInferenceEngine_impl
from .constants import CHECK_TIMEOUT

import typing as tp
import numpy as np
import asyncio

class GeneralInferenceEngine:
    """
        General inference engine for executing .engine files
    """

    def __init__(self, model_name: str, engine_path: str) -> None:
        """Constructs a new 'GeneralInferenceEngine' object
            
        Args:
          model_name: used for log message
          engine_path: path to engine file
        
        Raises:
          RuntimeError: If file not found
        """
        self.engine = GeneralInferenceEngine_impl(model_name, engine_path)

    @property
    def batch_size(self) -> int:
        """
        
        Returns:
          Engine batch size
        """
        return self.engine.batch_size()
    
    @property
    def np_input_shape(self) -> tp.Tuple[int]:
        """
        
        Returns:
          Input engine shape
        """
        return self.engine.np_input_shape()

    def predict(self, input_data: np.ndarray) -> tp.List[np.ndarray]:
        """Perform model inference.
        
        Args:
          input_data: numpy array with shape as np_input_shape and dtype float32
        
        Returns:
          List of predictions from output bindings
        
        Raises:
          RuntimeError: when input shape doesnt match or some CUDA error occurs
        """
        return self.engine.predict(input_data)

    def predict_async(self, input_data: np.ndarray) -> None:
        """Enqueue task for engine, but not wait for synchronization
        
        Args:
          input_data: numpy array with shape as np_input_shape and dtype float32

        Raises:
          RuntimeError: when input shape doesnt match or some CUDA error occurs
        """
        return self.engine.predict_async(input_data)

    def synchronize_async(self) -> tp.List[np.ndarray]:
        """Wait for executing of engine inference, which started by call of predict_async.
        
        Returns:
          List of predictions from output bindings

        Raises:
          RuntimeError: if some CUDA error occurs
        """
        return self.engine.synchronize_async()


class AsyncGeneralInferenceEngine(GeneralInferenceEngine):
    """Asynchronous version of GeneralInferenceEngine
    """

    def __init__(self, model_name: str, engine_path: str) -> None:
        super().__init__(model_name, engine_path)
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

    def predict(self, input_data: np.ndarray) -> tp.List[np.ndarray]:
        async def coro():
            self.engine.predict_callbacked(input_data)

            # send information to _notifier task, that execution started
            async with self.enqueue_condition:
                self.enqueue_condition.notify_all()

            # than wait until prediction is done
            async with self.predicted_condition:
                await self.predicted_condition.wait_for(lambda: not self.engine.is_inferencing())
            
            return self.engine.synchronize_callback()
        
        return coro()