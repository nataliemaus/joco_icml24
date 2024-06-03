import torch
from joco.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RosenbrockObjective(Objective):
    """Rosenbrock optimization task, original from
    https://www.sfu.ca/~ssurjano/rosen.html,
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
        assert output_dim == 2 * (input_dim - 1)

        self.default_budget = 2_000
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            lb=-5,
            ub=10,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 10

    def get_default_output_dim(self):
        return 18

    def x_to_y(self, x):
        x = x.to(device)
        x = x.squeeze()
        x_first = x[0:-1]
        x_sq = x_first**2
        x_next = x[1:]
        diffs = x_next - x_sq
        h_x = torch.cat((diffs, x_first))
        h_x = h_x.reshape(1, -1)
        self.num_calls += 1
        return h_x.detach().cpu()

    def y_to_score(self, y):
        y = y.to(device)
        y_first = y[0 : self.input_dim - 1]
        term1 = 100 * (y_first**2)
        y_next = y[self.input_dim - 1 :]
        term2 = (y_next - 1) ** 2
        reward = (term1 + term2).sum()
        reward = reward * -1  # make it a Maximization task
        return reward.item()
