import torch
from joco.tasks.objective import Objective
from joco.tasks.rover.rover_utils import create_large_domain, F_MAX


class RoverObjective(Objective):
    """Rover optimization task
    Goal is to find a policy for the Rover which
    results in a trajectory that moves the rover from
    start point to end point while avoiding the obstacles,
    thereby maximizing reward
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
        assert input_dim % 2 == 0  # input dim must be divisible by 2
        lb = -0.5 * 4 / input_dim
        ub = 4 / input_dim
        assert output_dim % 2 == 0  # output dim must be divisible by 2

        # Create rover domain
        self.domain = create_large_domain(
            n_points=input_dim // 2,
            n_samples=output_dim // 2,
            force_start=True,
            force_goal=False,
        )
        self.offset = F_MAX
        # rover oracle needs torch.double datatype
        self.tkwargs = {"dtype": torch.double}

        self.default_budget = 2_000

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 20

    def get_default_output_dim(self):
        return 1_000

    def x_to_y(self, x):
        traj_points = self.policy_to_trajectory(x)
        traj_tensor = torch.from_numpy(traj_points).float()
        traj_tensor = traj_tensor.reshape(1, -1)
        self.num_calls += 1
        return traj_tensor

    def y_to_score(self, y):
        traj_points = y.reshape(-1, 2)  # (out_dim,) --> (out_dim//2, 2)
        traj_points = traj_points.to(**self.tkwargs)
        traj_points = traj_points.cpu().numpy()
        reward = self.trajectory_to_reward(traj_points)
        return reward

    def policy_to_trajectory(self, policy):
        traj_points = self.domain.trajectory(policy.cpu().numpy())
        # traj_points.shape = (1000, 2) = (output_dim//2, 2)
        return traj_points

    def trajectory_to_reward(self, traj_points):
        cost = self.domain.traj_to_cost(traj_points)
        reward = -1 * cost
        reward = reward + self.offset
        reward_torch = torch.tensor(reward).to(**self.tkwargs)
        return reward_torch.item()
