from _ptxnn import GeneralInferenceEngine as GeneralInferenceEngine_impl

import typing as tp
import numpy as np

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

