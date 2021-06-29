from _ptxnn import set_severity as set_severity_impl
from _ptxnn import convert_yolo as convert_yolo_impl

from enum import Enum
import os

class Severity(Enum):
    kINTERNAL_ERROR = 0
    kERROR = 1
    kWARNING = 2
    kINFO = 3
    kVERBOSE = 4

def set_severity(severity: Severity) -> None:
    """Set severity of TensorRT logger

    Args:
      severity: logs level severity
    """
    set_severity_impl(severity.value)

def convert_yolo(cfg_path: str, layers_folder: str, names_path: str, 
            engine_path: str = 'yolo4_custom', mode: str = 'FP16', batchsize: int = 1) -> None:
    """Convert .bin files to YOLO .rt engine file

    Args:
      cfg_path: path to .cfg file with YOLO architecture description
      layers_folder: path to folder, which store .bin representation of each layer
      names_path: path to .names file with classes description
      mode: TensorRT weights mode
      batchsize: engine batch size

    Raises:
      FileNotFoundError: if one of pathes not exists
      ValueError:  if passed unsupported mode
    """
    mode = mode.upper()
    if mode not in ['FP16', 'FP32', 'DLA', 'INT8']:
        raise ValueError(f'Unsupported mode {mode}')

    for path in [cfg_path, layers_folder, names_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'No path {path}!')
    convert_yolo_impl(cfg_path, layers_folder, names_path, engine_path, mode, batchsize)