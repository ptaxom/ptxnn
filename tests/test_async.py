import sys
sys.path.append('..')
sys.path.append('../ptxnn')

import ptxnn
import asyncio
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Test async exection of model")
parser.add_argument('-m', '--model_path', dest='model_path', action='store',
                    required=True, type=str,
                    help='Path to other model engine file')
parser.add_argument('-y', '--yolo_path', dest='yolo_path', action='store',
                    required=True, type=str,
                    help='Path to YOLO engine file')
args = parser.parse_args()

ptxnn.set_severity(ptxnn.Severity.kINFO)
engine = ptxnn.AsyncGeneralInferenceEngine('model', args.model_path)
yolo = ptxnn.AsyncYoloEngine('yolo', args.yolo_path, 80, 0.3)

async def check_model(engine):
    a = np.zeros(engine.np_input_shape)
    for iter in range(10):
        result = await engine.predict(a)
        print(f'model({iter:02d}/10)', result)

async def check_yolo(engine: ptxnn.GeneralInferenceEngine):
    a = [np.zeros((512, 512, 3)) for _ in range(engine.batch_size)]
    for iter in range(5):
        result = await engine.predict_images(a)
        print(f'yolo({iter:02d}/5)', result)

async def background():
    for i in range(100):
        print('Iteration: ', i)
        await asyncio.sleep(0.02)

loop = asyncio.get_event_loop()
loop.run_until_complete(
    asyncio.gather(check_yolo(yolo), check_model(engine), background())
)
for task in asyncio.Task.all_tasks():
    task.cancel()
