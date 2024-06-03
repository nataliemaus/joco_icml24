import torch
from joco.tasks.objective import Objective

try:
    from pde import FieldCollection, PDE, ScalarField, UnitGrid
except ModuleNotFoundError:
    pass

# See original code for task here:https://github.com/wjmaddox/mtgp_sampler/blob/master/hogp_experiments/data.py
class PDEObjective(Objective):
    """PDE Task Described in
    section 4.4 of BO w/ High-Dimensional Outputs
    paper (https://arxiv.org/pdf/2106.12997.pdf)
    (See Optimizing PDEs Header)
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
        self.default_budget = 200
        assert input_dim >= 4
        assert output_dim == 64 * 64 * 2
        self.lower_bounds = [0.1, 0.1, 0.01, 0.01]
        self.upper_bounds = [5.0, 5.0, 5.0, 5.0]

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            lb=0,
            ub=1,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 32

    def get_default_output_dim(self):
        return 64 * 64 * 2

    def cfun(self, x, k=None):
        a = x[0].item()
        b = x[1].item()
        d0 = x[2].item()
        d1 = x[3].item()

        eq = PDE(
            {
                "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
            }
        )

        # initialize state
        grid = UnitGrid([64, 64])
        u = ScalarField(grid, a, label="Field $u$")
        v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$")
        state = FieldCollection([u, v])

        sol = eq.solve(state, t_range=20, dt=1e-3)
        sol_tensor = torch.stack(
            (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
        )
        sol_tensor[~torch.isfinite(sol_tensor)] = 1e5 * torch.randn_like(
            sol_tensor[~torch.isfinite(sol_tensor)]
        )
        return sol_tensor

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
        h_x = self.cfun(x)  # torch.Size([2, 64, 64])
        h_x = h_x.reshape(1, self.output_dim).float()
        return h_x.detach().cpu()  # torch.Size([1, 8192]) = (1, output_dim)

    def objective_fn(self, samples):
        # we want to minimize the variance across the boundaries
        sz = samples.shape[-1]
        weighting = (
            torch.ones(2, sz, sz, device=samples.device, dtype=samples.dtype) / 10
        )
        weighting[:, [0, 1, -2, -1], :] = 1.0
        weighting[:, :, [0, 1, -2, -1]] = 1.0

        weighted_samples = weighting * samples
        return weighted_samples.var(dim=(-1, -2, -3))

    def y_to_score(self, y):
        y = y.cuda()  # torch.Size([8192])
        samples = y.reshape(1, 2, 64, 64)
        loss = self.objective_fn(samples)  # loss = weighted variance
        reward = -1 * loss  # Minimization problem, * -1 for maximization
        return reward.item()
