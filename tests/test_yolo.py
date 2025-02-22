import sys
sys.path.append('..')
sys.path.append('../ptxnn')
import ptxnn

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Infer on test model")
parser.add_argument('-p', '--path', dest='path', action='store',
                    required=True, type=str,
                    help='Path to engine')
parser.add_argument('-i', '--input', dest='input_image', action='store',
                    required=True, type=str,
                    help='Path to test image')
args = parser.parse_args()

def render(image, detections):
    H, W = image.shape[:2]
    for det in detections:
        bbox = np.multiply(det[:4], [W, H, W, H]).astype('int')
        p1 = tuple(bbox[:2])
        p2 = tuple(bbox[2:])
        cv2.rectangle(image, p1, p2, (255, 0, 0), 1)

image = cv2.imread(args.input_image)
if image is None:
    raise RuntimeError(f'Couldnt load image {args.input_image}')

ptxnn.set_severity(ptxnn.Severity.kVERBOSE)
engine = ptxnn.YoloEngine("YOLO", args.path, 80, 0.3)


predict = engine.predict_images([image])
render(image, predict[0])
print(predict[0])
cv2.imshow('Predict', image)
cv2.waitKey()
