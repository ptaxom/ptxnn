import sys
sys.path.append('..')
sys.path.append('../ptxnn')
import ptxnn

import numpy as np
np.random.seed(42)
import argparse

parser = argparse.ArgumentParser(description="Infer on test model")
parser.add_argument('-y', '--yolo_path', dest='yolo_path', action='store',
                    required=True, type=str,
                    help='Path to YOLO engine file')
parser.add_argument('-m', '--model_path', dest='model_path', action='store',
                    required=True, type=str,
                    help='Path to other model engine file')
args = parser.parse_args()

ptxnn.set_severity(ptxnn.Severity.kVERBOSE)

class EngineWrapper:

    def __init__(self, engine):
        self.engine_obj = engine
        self.input_tensor = np.zeros(engine.np_input_shape, dtype=np.float32)
        self.predictions = []

    def ping(self):
        self.engine_obj.predict_async(self.input_tensor)

    def pong(self):
        prediction = self.engine_obj.synchronize_async()
        self.predictions.append(
            np.concatenate([x.reshape(-1) for x in prediction])
        )

    def valid(self):
        is_valid = True
        for i in range(len(self.predictions) - 1):
            is_valid &= np.allclose(self.predictions[i], self.predictions[i+1])
        return is_valid

engines = [
    EngineWrapper(ptxnn.GeneralInferenceEngine('GENERAL_MODEL', args.model_path)),
    EngineWrapper(ptxnn.GeneralInferenceEngine('YOLO', args.yolo_path))
]

for iteration in range(10):
    for i in range(len(engines)):
        engine = engines[(iteration + i) % 2]
        engine.ping()

    shift = int(np.random.random() > 0.5)
    for i in range(len(engines)):
        engine = engines[(iteration + i + shift) % 2]
        engine.pong()
    
passed = all([x.valid() for x in engines])
exit(not passed)