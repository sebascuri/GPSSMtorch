from .gpssm_vi import GPSSM, PRSSM, CBFSSM
from typing import Type


def get_model(model_: str) -> Type[GPSSM]:
    """Get Model."""
    if model_.lower() == 'prssm':
        return PRSSM
    elif model_.lower() == 'cbfssm':
        return CBFSSM
    else:
        raise NotImplementedError("{}".format(model_))
