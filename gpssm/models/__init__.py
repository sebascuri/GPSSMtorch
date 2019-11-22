from .gpssm_vi import GPSSM, PRSSM, CBFSSM
from .utilities import init_emissions, init_transitions, init_gps, init_recognition


def get_model(model_: str, dim_outputs: int, dim_inputs: int, dim_states: int = None,
              **kwargs) -> GPSSM:
    """Get Model."""
    dim_states = max(dim_states if dim_states is not None else dim_outputs, dim_outputs)

    gps = init_gps(dim_inputs, dim_states, **kwargs.pop('forward_gps', {}))  # type: ignore
    trans = init_transitions(dim_states, **kwargs.pop('transitions', {}))  # type: ignore
    emission = init_emissions(dim_outputs, **kwargs.pop('emissions', {}))  # type: ignore
    recognition = init_recognition(dim_outputs, dim_inputs, dim_states,  # type: ignore
                                   **kwargs.pop('recognition', {}))

    if model_.lower() == 'prssm':
        return PRSSM(forward_model=gps, transitions=trans, emissions=emission,
                     recognition_model=recognition, **kwargs)
    elif model_.lower() == 'cbfssm':
        return CBFSSM(forward_model=gps, transitions=trans, emissions=emission,
                      recognition_model=recognition, **kwargs)
    else:
        raise NotImplementedError("{}".format(model_))
