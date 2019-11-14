from .dataset import Actuator, BallBeam, Drive, Dryer, Flutter, GasFurnace
from typing import Type, Union


def get_dataset(dataset_: str) -> Union[Type[Actuator], Type[BallBeam], Type[Drive],
                                        Type[Dryer], Type[Flutter], Type[GasFurnace]]:
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
    else:
        raise NotImplementedError("{}".format(dataset_))
