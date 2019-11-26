from .ssm import SSM, PRSSM, CBFSSM
from .utilities import init_emissions, init_transitions, init_dynamics, init_recognition


def get_model(model_: str, dim_outputs: int, dim_inputs: int, dim_states: int = None,
              **kwargs) -> SSM:
    """Get GPSSM Model."""
    dim_states = max(dim_states if dim_states is not None else dim_outputs, dim_outputs)

    f_config = kwargs.pop('forward', {})
    forward_model = init_dynamics(dim_inputs + dim_states, dim_states, **f_config
                                  )  # type: ignore

    backward_model = init_dynamics(dim_inputs + dim_states, dim_states - dim_outputs,
                                   **kwargs.pop('backward', f_config)
                                   )  # type: ignore

    transitions = init_transitions(dim_states,
                                   **kwargs.pop('transitions', {}))  # type: ignore
    emission = init_emissions(dim_outputs,
                              **kwargs.pop('emissions', {}))  # type: ignore
    recognition = init_recognition(dim_outputs, dim_inputs, dim_states,  # type: ignore
                                   **kwargs.pop('recognition', {}))

    if model_.lower() == 'prssm':
        return PRSSM(forward_model=forward_model,
                     transitions=transitions, emissions=emission,
                     recognition_model=recognition,
                     **kwargs)
    elif model_.lower() == 'cbfssm-half':
        return CBFSSM(forward_model=forward_model,
                      transitions=transitions, emissions=emission,
                      recognition_model=recognition,
                      **kwargs)
    elif model_.lower() == 'cbfssm':
        return CBFSSM(forward_model=forward_model, backward_model=backward_model,
                      transitions=transitions, emissions=emission,
                      recognition_model=recognition,
                      **kwargs)
    else:
        raise NotImplementedError("{}".format(model_))
