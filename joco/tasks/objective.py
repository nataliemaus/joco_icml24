# Parent class for differnet Objectives/ tasks
import numpy as np
import torch


class Objective:
    """Base class for any optimization task
    class supports oracle calls and tracks
    the total number of oracle class made during
    optimization
    """

    def __init__(
        self,
        num_calls=0,
        input_dim=None,
        output_dim=None,
        lb=None,
        ub=None,
        unique_run_id=None,
        use_custom_y_compression_model=False,
    ):
        # unique run id for current run, used for logging outputs
        self.unique_run_id = unique_run_id
        # use customized model to compress y (i.e. conv net for images)
        self.use_custom_y_compression_model = use_custom_y_compression_model
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # input dim (x)
        if input_dim is None:
            input_dim = self.get_default_input_dim()
        self.input_dim = input_dim
        # output dim (y)
        if output_dim is None:
            output_dim = self.get_default_output_dim()
        self.output_dim = output_dim
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub

    def __call__(self, xs):
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, input_dim) enumerable tye of length equal to batch size (bsz), each item in enumerable type must be a float tensor of shape (input_dim,) (each is a vector in input search space).

        Returns:
            tensor: (bsz, 1) float tensor giving reward obtained by passing each x in xs into f(x).

        """
        if type(xs) is np.ndarray:
            xs = torch.from_numpy(xs).float()
        ys = self.xs_to_ys(xs)
        scores = self.ys_to_scores(ys)
        return scores

    def xs_to_ys(self, xs):
        """Function xs_to_ys defines batched function h(x) where we want to optimize f(x) = g(h(x)).

        Args:
            xs (enumerable):  enumerable tye of length equal to batch size (bsz), each item in enumerable type must be a float tensor of shape (input_dim,) (each is a vector in input search space).

        Returns:
            tensor: (bsz, output_dim) float tensor giving intermediate output y obtained by passing each x in xs into h(x).

        """
        ys = []
        for x in xs:
            ys.append(self.x_to_y(x))
        return torch.cat(ys)

    def ys_to_scores(self, ys):
        """Function ys_to_scores defines batched function g(y) where we want to optimize f(x) = g(h(x)) = g(y)

        Args:
            ys (enumerable):  enumerable tye of length equal to batch size (bsz), each item in enumerable type must be a float tensor of shape (output_dim,) (each is a vector in intermediate output space).

        Returns:
            tensor: (bsz, 1) float tensor giving reward obtained by each y in input ys.

        """
        scores = []
        for y in ys:
            scores.append(self.y_to_score(y))
        return torch.tensor(scores).unsqueeze(-1)

    def x_to_y(self, x):
        """Function x_to_y defines function h(x) where we want to optimize f(x) = g(h(x)). This method should also increment self.num_calls by one.

        Args:
            x (tensor): (1, input_dim) float tensor giving vector in input search space.

        Returns:
            tensor: (1, output_dim) float tensor giving vector in intermediate output space.

        """
        raise NotImplementedError(
            "Must implement x_to_y() specific to desired optimization task"
        )

    def y_to_score(self, y):  # output dim to score
        """Function x_to_score defines function g(y) where we want to optimize f(x) = g(h(x)) = g(y)

        Args:
            y (tensor): (output_dim,) float tensor giving vector in intermediate output space.

        Returns:
            float: reward value obtained by passing y into utility function g(y).

        """
        raise NotImplementedError(
            "Must implement y_to_score() specific to desired optimization task"
        )

    def get_default_input_dim(self):
        """Function get_default_input_dim returns the default dimensionality of input search space.

        Args:
            None

        Returns:
            int: default dimensionality of input search space.

        """
        raise NotImplementedError(
            "Must implement get_default_input_dim() specific to desired optimization task"
        )

    def get_default_output_dim(self):
        """Function get_default_output_dim returns the default dimensionality of intermediate output space.

        Args:
            None

        Returns:
            int: default dimensionality of output space.

        """
        raise NotImplementedError(
            "Must implement get_default_output_dim() specific to desired optimization task"
        )
