try:
    import mcdnn
except:
    import sys
    sys.path.append('../build/')
    import mcdnn

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Infer on test model")
parser.add_argument('-m', '--model', dest='model', action='store',
                    required=False, default='TestModel', type=str,
                    help='Model name used for logging')
parser.add_argument('-p', '--path', dest='path', action='store',
                    required=True, type=str,
                    help='Path to engine')
parser.add_argument('-d', '--dim', dest='dim', action='store',
                    required=False, type=int, default=256,
                    help='Input dim, in case of square form')
parser.add_argument('-b', '--batch_size', dest='batchsize', action='store',
                    required=False, type=int, default=1,
                    help='Input dim, in case of square form')
parser.add_argument('-i', '--input', dest='input_image', action='store',
                    required=True, type=str,
                    help='Path to test image')
args = parser.parse_args()

image = cv2.imread(args.input_image)
if image is None:
    raise RuntimeError(f'Couldnt load image {args.input_image}')

engine = mcdnn.GeneralInferenceEngine(args.model, args.path)

sz = args.dim
resize_img = cv2.resize(image, (sz, sz), interpolation=cv2.INTER_CUBIC)
type_img = resize_img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)

predict = engine.predict(type_img)
print(predict)
print(np.sum(predict[0] ** 2))
