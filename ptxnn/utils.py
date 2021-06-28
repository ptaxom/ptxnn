from _ptxnn import set_severity as set_severity_impl

from enum import Enum

class Severity(Enum):
    kINTERNAL_ERROR = 0
    kERROR = 1
    kWARNING = 2
    kINFO = 3
    kVERBOSE = 4

def set_severity(severity: Severity) -> None:
    """
    Set severity of TensorRT logger
    """
    set_severity_impl(severity.value)