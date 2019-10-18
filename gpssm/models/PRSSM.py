"""Implementation of PR-SSM algorithm."""

# class SVIMarkov(object):
#     pass
#
#
# class PRSSM(SVIMarkov):
#     def __init__(self, dim_x: int,
#                  emission_model: callable,
#                  transition_model: callable,
#                  gp_model: callable):
#         self.px0 = {
#             'mean': torch.nn.Parameter(torch.zeros(dim_x), requires_grad=True),
#             'cholesky': torch.nn.Parameter(torch.eye(dim_x), requires_grad=True),
#         }
#
#         self.qx0 = {
#             'mean': torch.nn.Parameter(torch.zeros(dim_x), requires_grad=True),
#             'cholesky': torch.nn.Parameter(torch.eye(dim_x), requires_grad=True),
#         }
#
#         self.emission = emission_model
#         self.transition = transition_model
#         self.gp = gp_model
#
#     def log_likelihood(self, output_sequence, input_sequence):
#         sequence_len = output_sequence.shape[0]
#
#         state = MultivariateNormal(self.qx0['mean'], scale_tril=self.qx0['cholesky'])
#
#         log_lik = torch.zeros(1)
#         for t in range(sequence_len):
#             output_dist = self.emission(state)
#             f_t = self.gp(state, input_sequence[t])
#             next_state_dist = self.transition(f_t)
#             state = next_state_dist.rsample()
#
#             log_lik += output_dist.log_prob(output_sequence[t])
#
#         return log_lik
#
#     def kl(self):
#         return self.gp.kl_divergence()
