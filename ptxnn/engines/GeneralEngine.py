from _ptxnn import GeneralInferenceEngine as GeneralInferenceEngine_impl

import typing as tp

class GeneralInferenceEngine:
    """
        General inference engine for executing .engine files
    """

    def __init__(self, model_name: str, engine_path: str) -> None:
        """
            model_name: str
            Will be used for log message
            engine_path: str
            Engine file
        """
        self.engine = GeneralInferenceEngine_impl(model_name, engine_path)
