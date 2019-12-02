"""Python Script Template."""
import matplotlib.pyplot as plt
import torch
from gpssm.utilities import Experiment, load
from gpssm.models import SSM, get_model
from gpssm.dataset import get_dataset
from gpssm.utilities import approximate_with_normal
from gpssm.plotters import plot_2d

dim_states = 4
dataset = get_dataset('robomovesimple')
model = get_model('prssm', dim_states=dim_states,
                  dim_inputs=dataset.dim_inputs, dim_outputs=dataset.dim_outputs)
train_set = dataset(train=True, sequence_length=300, sequence_stride=1)
test_set = dataset(train=False, sequence_length=300, sequence_stride=1)


with torch.no_grad():
    inputs, outputs, states = train_set[0]
    pred_outputs, _ = model.forward(outputs.unsqueeze(0), inputs.unsqueeze(0))

    pred_outputs = approximate_with_normal(pred_outputs)
    mean = pred_outputs.loc.detach().numpy()

    fig = plot_2d(mean[0].T, true_outputs=outputs.numpy().T)
    fig.show()
