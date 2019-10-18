"""Project main runner file."""

# from gpssm.dataset.dataset import KinkFunction
# from torch.utils.data import DataLoader
# import pyro
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO
# pyro.clear_param_store()
#
# dataset = KinkFunction(train=True, sequence_length=1)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
#
# model = ...
# guide = ...
# optimizer = Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
#
# svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
#
# for idx, (inputs, outputs, states) in enumerate(data_loader):
#     print(idx, inputs.shape, outputs.shape, states.shape)
#
#     svi.step(data)
#     model(inputs, outputs, )
