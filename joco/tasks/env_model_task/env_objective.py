import math

import torch
from joco.tasks.objective import Objective

# See original code for task here: https://github.com/wjmaddox/mtgp_sampler/blob/master/hogp_experiments/data.py
class EnvObjective(Objective):
    """Environmental Pollutants Task Described in
    section 4.4 of BO w/ High-Dimensional Outputs
    paper (https://arxiv.org/pdf/2106.12997.pdf)
    Same as Env objective from BOCF paper, modified
    to be higher-dimensional
    """

    def __init__(
        self,
        input_dim=None,
        output_dim=None,
        **kwargs,
    ):
        if input_dim is None:
            input_dim = self.get_default_input_dim()
        if output_dim is None:
            output_dim = self.get_default_output_dim()
        # Note: Must have output dim is 12 or a power of 2
        if not (output_dim == 12):
            assert (
                output_dim & (output_dim - 1) == 0
            ), "Output dim must either be 12 or a power of 2"
        # Only the first four input dims matter but we can add extra dims and learn to ignore them
        self.default_budget = 1_000
        assert input_dim >= 4, "At least 4 input dims are required"
        M0 = torch.tensor(10.0).float().cuda()
        D0 = torch.tensor(0.07).float().cuda()
        L0 = torch.tensor(1.505).float().cuda()
        tau0 = torch.tensor(30.1525).float().cuda()
        if output_dim == 12:
            self.s_size = 3
            self.t_size = 4
        else:
            # Otherwise s and t sizes are root output dim
            self.s_size = int(output_dim**0.5)
            self.t_size = int(output_dim**0.5)
            # Make sure output dim is indeed a power of 2
            assert output_dim == self.s_size * self.t_size
        if self.s_size == 3:
            S = torch.tensor([0.0, 1.0, 2.5]).float().cuda()
        else:
            S = torch.linspace(0.0, 2.5, self.s_size).float().cuda()
        if self.t_size == 4:
            T = torch.tensor([15.0, 30.0, 45.0, 60.0]).float().cuda()
        else:
            T = torch.linspace(15.0, 60.0, self.t_size).float().cuda()

        self.Sgrid, self.Tgrid = torch.meshgrid(S, T)
        self.c_true = self.env_cfun(self.Sgrid, self.Tgrid, M0, D0, L0, tau0)
        # Bounds used to unnormalize x (optimize in 0 to 1 range for all)
        self.lower_bounds = [7.0, 0.02, 0.01, 30.010]
        self.upper_bounds = [13.0, 0.12, 3.00, 30.295]
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            lb=0,
            ub=1,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 15

    def get_default_output_dim(self):
        return 16

    def env_cfun(self, s, t, M, D, L, tau):
        c1 = M / torch.sqrt(4 * math.pi * D * t)
        exp1 = torch.exp(-(s**2) / 4 / D / t)
        term1 = c1 * exp1
        c2 = M / torch.sqrt(4 * math.pi * D * (t - tau))
        exp2 = torch.exp(-((s - L) ** 2) / 4 / D / (t - tau))
        term2 = c2 * exp2
        term2[torch.isnan(term2)] = 0.0
        return term1 + term2

    def x_to_y(self, x):
        self.num_calls += 1
        x = x.squeeze().cuda()
        x = x[0:4]
        # Unnormalize each dim of x
        for i in range(4):
            x[i] = (
                x[i] * (self.upper_bounds[i] - self.lower_bounds[i])
            ) + self.lower_bounds[i]
        # compute h_x
        h_x = self.env_cfun(self.Sgrid, self.Tgrid, *x)
        h_x = h_x.reshape(1, self.output_dim)
        return h_x.detach().cpu()

    def y_to_score(self, y):
        y = y.cuda()
        y = y.unsqueeze(-1).reshape(*y.shape[:-1], self.s_size, self.t_size)
        sq_diffs = (y - self.c_true).pow(2)
        reward = sq_diffs.sum(dim=(-1, -2)).mul(-1.0)
        return reward.item()
