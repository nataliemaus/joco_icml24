import math

import torch
from joco.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LangermannObjective(Objective):
    """Langermann optimization task, original from
    https://www.sfu.ca/~ssurjano/langer.html,
    adapted to a composite function by
    BO for Composite Functions Paper
    (https://arxiv.org/abs/1906.01537 see Appendix D)
    Designed as a minimization task so we multiply by -1
    to obtain a maximization task
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
        assert input_dim % 2 == 0
        assert output_dim % 5 == 0
        self.A = torch.tensor(
            [[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]
        ).float()  # (2,5) defaults
        # A = (2,5) by default, need A = (input_dim, output_dim)
        self.A = self.A.repeat(
            input_dim // 2, output_dim // 5
        )  # (input_dim, output_dim)
        self.A = self.A.to(device)
        # A = (5,) by default, need c = (output_dim,)
        self.c = torch.tensor([1, 2, 5, 2, 3]).float()  # (5,)
        self.c = self.c.repeat(
            output_dim // 5,
        )  # (output_dim,)
        self.c = self.c.to(device)
        # grab PI
        self.pi = math.pi
        # default n init
        self.default_budget = 4_000

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            lb=0,
            ub=10,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 16

    def get_default_output_dim(self):
        return 60

    def x_to_y(self, x):
        x = x.to(device)
        x_repeated = torch.cat([x.reshape(self.input_dim, 1)] * self.output_dim, -1)
        h_x = ((x_repeated - self.A) ** 2).sum(0)
        h_x = h_x.reshape(1, -1)
        self.num_calls += 1
        return h_x.detach().cpu()

    def y_to_score(self, y):
        y = y.to(device)
        reward = self.c * torch.exp((-1 * y) / self.pi) * torch.cos(self.pi * y)
        reward = reward.sum()
        return reward.item()
