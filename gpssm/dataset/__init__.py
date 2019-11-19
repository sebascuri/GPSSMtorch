from .dataset import Actuator, BallBeam, Drive, Dryer, Flutter, GasFurnace, Tank, \
    RoboMoveSimple, RoboMove, Sarcos, KinkFunction, NonLinearSpring, Dataset
from typing import Type


def get_dataset(dataset_: str) -> Type[Dataset]:
    """Get Dataset."""
    if dataset_.lower() == 'actuator':
        return Actuator
    elif dataset_.lower() == 'ballbeam':
        return BallBeam
    elif dataset_.lower() == 'drive':
        return Drive
    elif dataset_.lower() == 'dryer':
        return Dryer
    elif dataset_.lower() == 'flutter':
        return Flutter
    elif dataset_.lower() == 'gasfurnace':
        return GasFurnace
    elif dataset_.lower() == 'tank':
        return Tank
    elif dataset_.lower() == 'sarcos':
        return Sarcos
    elif dataset_.lower() == 'kinkfunction':
        return KinkFunction
    elif dataset_.lower() == 'robomove':
        return RoboMove
    elif dataset_.lower() == 'robomovesimple':
        return RoboMoveSimple
    elif dataset_.lower() == 'spring':
        return NonLinearSpring
    else:
        raise NotImplementedError("{}".format(dataset_))
