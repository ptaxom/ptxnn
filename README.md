# ptxnn
ptxnn is python wrapper of [tkDNN](https://github.com/ceccocats/tkDNN) library. It have almost all features of tkDNN, but focused on building efficient unified interface of multiple TensorRT engines, including YOLOv4, Scaled YOLOv4.

# Code samples
Usage with classification model:
```python3
import ptxnn
# set maximum TensorRT verbosity, so you will all message
ptxnn.set_severity(ptxnn.Severity.kVERBOSE)
# Loading model
engine = ptxnn.GeneralInferenceEngine('ResNet-50', '/path/to/resnet50.engine')
images = [cv2.imread(image) for image in input_files]
images = np.concatenate([cv2.resize(image, (224, 224))[np.newaxis, ...] for image in images], axis=0) / 255
input_tensor = np.transpose(frames, (0, 3, 1, 2)).float32()
result = engine.predict(input_tensor)[0]
print(result)
```

Usage with YOLO model
```python3
import ptxnn

ptxnn.set_severity(ptxnn.Severity.kVERBOSE)
engine = ptxnn.YoloEngine("YOLO", '/path/to/tkDNN/yolo4.rt', 80, 0.3)
images = [cv2.imread(image) for image in input_files]
# Yes, it's that simple!
result = engine.predict_images(images)
print(result)
```

It also support async API, so you can use it in your Fast-API app or to synchronize multiple models.
```python3
engine1 = ptxnn.AsyncGeneralInferenceEngine(...)
engine2 = ptxnn.AsyncYoloEngine(...)

async def api_handler(images):
    task = asyncio.gather(engine1.predict(images), engine2.predict(images))
    result = await task
    return result
```

# Installation
Make sure, that can you build original tkDNN project. Than you can easily install ptxnn
```bash
pip3 install pybind11[global] setuptools build
python3 -m build -w -n
pip3 install dist/*.whl
```