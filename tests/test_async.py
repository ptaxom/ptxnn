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
args = parser.parse_args()

ptxnn.set_severity(ptxnn.Severity.kINFO)
engine = ptxnn.AsyncGeneralInferenceEngine('model', args.model_path)

async def check_model(engine):
    a = np.zeros(engine.np_input_shape)
    result = await engine.predict(a)
    print(result)

async def background():
    for i in range(20):
        print('Iteration: ', i)
        await asyncio.sleep(0.02)

loop = asyncio.get_event_loop()
loop.run_until_complete(
    asyncio.gather(check_model(engine), background())
)
for task in asyncio.Task.all_tasks():
    task.cancel()
