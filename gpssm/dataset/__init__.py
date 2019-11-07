from .dataset import Actuator, BallBeam, Drive, Dryer, Flutter, GasFurnace, Dataset


def get_dataset(dataset_: str) -> Dataset:
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
