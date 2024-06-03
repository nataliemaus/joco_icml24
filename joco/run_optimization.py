import json
import os
import time
import uuid

import cma

import gpytorch
import torch
from gpytorch.mlls import PredictiveLogLikelihood

from joco.models.gp import GPModel
from joco.models.gp_dkl import GPModelDKL
from joco.models.gp_shared_dkl import GPModelSharedDKL
from joco.models.simple_nn import DenseNetwork
from joco.models.update_models import update_model, update_surrogate_models_joco
from joco.utils.set_seed import set_seed
from joco.utils.turbo import (
    generate_batch,
    generate_batch_joco,
    TurboState,
    update_state_unconstrained,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimize:
    """
    Optimize high-dim input and output task
    Args:
        task_id: string id for optimization task
        seed: Random seed to be set. If None, no particular random seed is set
        max_n_oracle_calls: Max number of oracle calls allowed (budget). If -1, use default for task
        learning_rte: Learning rate for model updates
        acq_func: Acquisition function, must be either ei or ts (ei-->Expected Imporvement, ts-->Thompson Sampling)
        bsz: Acquisition batch size
        num_initialization_points: int, Number evaluated data points used to optimization initialize run. If -1, use default for task
        init_n_epochs: Number of epochs to train the surrogate model for on initial data before optimization begins
        num_update_epochs: Number of epochs to update the model(s) for on each optimization step
        gp_hidden_dims: tuple giving hidden dims for GP Deep Kernel, if not specified will be (input_dim, input_dim) by default
        gp_y_model_hidden_dims: tuple giving h dims for additional GPDKL Model for modeling g(y) for joco only, if not specified will use default
        optimizer: string must be one of "joco", "turbo", or "vanilla" (indicating vanilla BO)
        max_lookback: int, max N train data points to update on in each iteration of BO (will update on most recent N points)
        use_dkl: bool, if true use GP with a deep kernel to compress the input space
        verbose: bool, if True print progress update on each iteration of BO
        save_run_data: bool, if True save all data from optimization run locally after run finishes
        train_bsz: int, batch size upsed for updating surrogate model(s) on data
        use_tr: bool, if True use trust regions for JoCo (only relevant when running JoCo)
        input_dim: int, input search space dimension for optimizatoin task. If None provided, we use default for task
        output_dim: int, intermediate output dimension for optimizatoin task. If None provided, we use default for task
        update_jointly: bool, if True update jointly with JoCo (set to false to ablate the joint updates)
        propegate_uncertainty_x: bool, if True, propagate uncertainty through gpx during acqusition (use False for JoCo ablation)
        propegate_uncertainty_y: bool, if True, propagate uncertainty through gpy during acqusition (use False for JoCo ablation)
        rand_proj_baseline: bool, if True, run joco with random projections to compress dims instead of NNs 
    """

    def __init__(
        self,
        task_id: str = "langermann",
        seed: int = 0,
        max_n_oracle_calls: int = -1,
        learning_rte: float = 0.001,
        acq_func: str = "ts",
        bsz: int = 10,
        num_initialization_points: int = -1,
        init_n_epochs: int = 30,
        num_update_epochs: int = 1,
        gp_hidden_dims=(),
        gp_y_model_hidden_dims=(),
        optimizer: str = "joco",
        max_lookback: int = 20,
        use_dkl: bool = True,
        verbose: bool = True,
        save_run_data: bool = True,
        train_bsz: int = 64,
        use_tr: bool = True,
        input_dim=None,
        output_dim=None,
        update_jointly: bool = True,
        propegate_uncertainty_x: bool = True,
        propegate_uncertainty_y: bool = True,
        rand_proj_baseline: bool = False,
        load_random_init_train_data: bool = True,
    ):
        unique_run_id = str(uuid.uuid4().hex)
        self.unique_run_id = unique_run_id
        # initialize latent space objective (self.objective) for particular task
        self.set_objective_class(
            task_id=task_id,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        # In the case input and output dims are not provided, we use defaults associated w/ objective
        input_dim = self.objective.input_dim
        output_dim = self.objective.output_dim
        use_custom_y_compression_model = self.objective.use_custom_y_compression_model

        if not gp_hidden_dims:
            gp_hidden_dims = (input_dim // 2, input_dim // 2)
        if not gp_y_model_hidden_dims:
            if output_dim >= 256:
                gp_y_model_hidden_dims = (
                    min(output_dim, 256),
                    min(output_dim // 2, 128),
                    32,
                )
            else:
                first_dim = max(output_dim // 2, 32)
                first_dim = min(first_dim, output_dim)
                gp_y_model_hidden_dims = (first_dim, 8)
        if max_n_oracle_calls == -1:
            max_n_oracle_calls = self.objective.default_budget
        if num_initialization_points == -1:
            # if none given, use 10% of the budget
            num_initialization_points = max_n_oracle_calls // 10

        self.method_args = {}
        self.method_args["init"] = locals()
        del self.method_args["init"]["self"]
        self.method_args = {
            k: v
            for method_dict in self.method_args.values()
            for k, v in method_dict.items()
        }
        # use lists to hold n calls vs best scores throughout
        self.log_n_oracle_calls = []
        self.log_best_scores = []
        self.log_best_prompts = []  # for stable diffusion tasks
        self.best_prompts = []  # for stable diffusion tasks
        self.save_run_data = save_run_data
        self.verbose = verbose
        self.train_bsz = train_bsz
        self.use_custom_y_compression_model = use_custom_y_compression_model
        self.seed = seed
        self.task_id = task_id
        self.max_n_oracle_calls = max_n_oracle_calls
        self.num_initialization_points = num_initialization_points
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bsz = bsz
        self.init_n_epochs = init_n_epochs
        self.num_update_epochs = num_update_epochs
        self.initial_model_training_complete = False
        self.learning_rte = learning_rte
        self.gp_hidden_dims = gp_hidden_dims
        self.gp_y_model_hidden_dims = gp_y_model_hidden_dims
        self.acq_func = acq_func
        self.optimizer = optimizer
        self.rand_proj_baseline = rand_proj_baseline
        if rand_proj_baseline:
            assert optimizer == "joco"

        self.max_lookback = max_lookback
        if optimizer == "vanilla":
            self.use_tr = False
        elif optimizer == "turbo":
            self.use_tr = True
        elif optimizer == "joco":
            self.use_tr = use_tr
        self.use_dkl = use_dkl
        self.update_jointly = update_jointly
        self.propegate_uncertainty_x = propegate_uncertainty_x
        self.propegate_uncertainty_y = propegate_uncertainty_y
        set_seed(self.seed)
        # initialize train data for particular task
        if load_random_init_train_data:
            # keep option to set to false in case we pre-compute and set train data to save time (i.e. in analyze_model_performance.py)
            self.load_train_data()
            # check for correct initialization of train data:
            assert torch.is_tensor(
                self.train_y
            ), f"load_train_data() must set self.train_y to a torch tensor of ys,\
                instead got self.train_y of type {type(self.train_y)}"
            assert torch.is_tensor(
                self.train_scores
            ), f"load_train_data() must set self.train_scores to a torch tensor of scores,\
                instead got self.train_scores of type {type(self.train_scores)}"
            assert (self.train_y.shape[0] == self.num_initialization_points) and (
                self.train_y.shape[1] == self.output_dim
            ), f"load_train_data() must initialize self.train_y with dims \
                (self.num_initialization_points,self.output_dim)\
                =({self.num_initialization_points},{self.output_dim}),\
                instead got self.train_y with dims {self.train_y.shape}"
            assert (self.train_x.shape[0] == self.num_initialization_points) and (
                self.train_x.shape[1] == self.input_dim
            ), f"load_train_data() must initialize self.train_x with dims \
                (self.num_initialization_points, self.input_dim)\
                =({self.num_initialization_points},{self.input_dim}),\
                instead got self.train_x with dims {self.train_x.shape}"
            assert (self.train_scores.shape[0] == self.num_initialization_points) and (
                self.train_scores.shape[1] == 1
            ), f"load_train_data() must initialize self.train_scores with dims \
                (self.num_initialization_points, 1), instead got \
                self.train_scores with dims {self.train_scores.shape}"

    def set_objective_class(self, task_id, input_dim, output_dim):
        if task_id in ["dog", "sportscar", "aircraft"]:
            self.set_stable_diffusion_task_objective(task_id, input_dim, output_dim)
        elif task_id in ["llama", "gpt2", "opt", "falcon"]:
            self.set_llm_task_objective(task_id, input_dim, output_dim)
        else:
            self.set_synthetic_task_objective(task_id, input_dim, output_dim)

    def set_synthetic_task_objective(self, task_id, input_dim, output_dim):
        if task_id == "gp1":
            from joco.tasks.gp_generated_tasks.gp1_objective import GPObjective1 as Obj
        elif task_id == "gp2":
            from joco.tasks.gp_generated_tasks.gp2_objective import GPObjective2 as Obj
        elif task_id == "langermann":
            from joco.tasks.langermann.langermann_objective import (
                LangermannObjective as Obj,
            )
        elif task_id == "rosenbrock":
            from joco.tasks.rosenbrock.rosenbrock_objective import (
                RosenbrockObjective as Obj,
            )
        elif task_id == "rover":
            from joco.tasks.rover.rover_objective import RoverObjective as Obj
        elif task_id == "env":
            from joco.tasks.env_model_task.env_objective import EnvObjective as Obj
        elif task_id == "mnist":
            from joco.tasks.mnist.mnist_objective import MNISTObjective as Obj
        elif task_id == "pde":
            from joco.tasks.pde_task.pde_objective import PDEObjective as Obj
        else:
            assert 0, f"provided task_id: {task_id} was not recognized"
        self.objective = Obj(
            unique_run_id=self.unique_run_id,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def set_stable_diffusion_task_objective(self, task_id, input_dim, output_dim):
        if task_id == "sportscar":
            from joco.tasks.stable_diffusion_tasks.sportscar_objective import (
                GenerateSportsCars as Obj,
            )
        elif task_id == "dog":
            from joco.tasks.stable_diffusion_tasks.dog_objective import (
                GenerateDogsPrepending as Obj,
            )
        elif task_id == "aircraft":
            from joco.tasks.stable_diffusion_tasks.aircraft_objective import (
                GenerateAircraftsPrepending as Obj,
            )
        self.objective = Obj(
            unique_run_id=self.unique_run_id,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def set_llm_task_objective(self, task_id, input_dim, output_dim):
        if task_id == "llama":
            from joco.tasks.llm_tasks.llama.llama_objective import LlamaObjective as Obj
        elif task_id == "gpt2":
            from joco.tasks.llm_tasks.gpt2.gpt2_objective import GPT2Objective as Obj
        elif task_id == "opt":
            from joco.tasks.llm_tasks.opt.opt_objective import OPTObjective as Obj
        elif task_id == "falcon":
            from joco.tasks.llm_tasks.falcon.falcon_objective import (
                FalconObjective as Obj,
            )
        self.objective = Obj(
            unique_run_id=self.unique_run_id,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    def load_train_data(self):
        """Load in or randomly initialize self.num_initialization_points
        total initial data points to kick-off optimization
        """
        self.train_x = self.get_random_x_points(n_points=self.num_initialization_points)
        self.train_y = self.objective.xs_to_ys(self.train_x)
        self.train_scores = self.objective.ys_to_scores(self.train_y)

    def get_random_x_points(self, n_points):
        """Get random data points in X search space"""
        if (self.objective.lb is not None) and (
            self.objective.ub is not None
        ):  # convert to random uniform in bounds if bounds given
            random_x = torch.rand(n_points, self.input_dim)  # 0 to 1 uniform
            random_x = (
                random_x * (self.objective.ub - self.objective.lb) + self.objective.lb
            )  # convert to uniform in bounds
        else:
            random_x = torch.randn(
                n_points, self.input_dim
            )  # random normal dist (important to include negatives!)

        return random_x

    def update_best(
        self,
    ):
        self.best_score = self.train_scores.max().item()
        self.best_x = self.train_x[self.train_scores.argmax()]
        self.best_y = self.train_y[self.train_scores.argmax()]
        # if this is a diffusion model prompt optimization task, we want to log the best prompt not the best word embedding
        if self.task_id in ["sportscar", "dog", "aircraft"]:
            self.best_x = self.best_x.reshape(
                -1, self.objective.n_tokens, self.objective.word_embedding_dim
            )
            self.best_x = self.objective.proj_word_embedding(self.best_x.to(device))[0][
                0
            ]

    def log_data_on_each_loop(self):
        self.log_n_oracle_calls.append(self.objective.num_calls)
        self.log_best_scores.append(self.best_score)
        if self.task_id in ["sportscar", "dog", "aircraft"]:
            self.log_best_prompts.append(self.best_x)
        if self.verbose:
            print(
                f"\nNumber of oracle calls:{self.objective.num_calls}, Best score:{self.best_score}"
            )
        self.dict_log = {
            # "best_x":self.best_x.detach().cpu(), # Saving these large tensors eats up space and is unnessary for plotting, etc.
            # "best_y":self.best_y.detach().cpu(),
            "n_oracle_calls": self.objective.num_calls,
            "best_score": self.best_score,
            "total_run_time": time.time() - self.start_opt_time,
            "method_args": self.method_args,
            "log_nums_oracle_calls": self.log_n_oracle_calls,
            "log_best_scores": self.log_best_scores,
            "log_best_prompts": self.log_best_prompts,
        }
        """ if this is a diffusion model prompt optimization task, we want to log all the best prompts found along the way
            (so we can re-generate images from them to look at later)"""
        if self.task_id in ["sportscar", "dog", "aircraft"]:
            self.best_prompts.append(self.best_x)
            self.dict_log["best_prompts"] = self.best_prompts

    def init_cmaes(self):
        # https://botorch.org/tutorials/optimize_with_cmaes
        x0 = self.train_x[self.train_scores.argmax()]
        x0 = x0.detach().cpu().numpy().squeeze()
        sigma0 = self.train_y.squeeze().std().item()
        if (self.objective.ub is not None) and (self.objective.lb is not None):
            self.es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=sigma0,
                inopts={
                    "bounds": [self.objective.lb, self.objective.ub],
                    "popsize": self.bsz,
                },
            )
        else:
            self.es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=sigma0,
                inopts={"popsize": self.bsz},
            )

    def initialize_trust_region(self):
        self.tr_state = TurboState(
            dim=self.objective.input_dim,
            batch_size=self.bsz,
        )

    def run_optimization(self):
        """Main optimization loop"""
        self.start_opt_time = time.time()
        # initialize trust reigon
        self.initialize_trust_region()
        # grab best point found so far (in init data)
        self.update_best()
        # initialize es if running cma-es
        if self.optimizer == "cma-es":
            self.init_cmaes()
        # log init data
        self.log_data_on_each_loop()
        # initialize surrogate model(s)
        self.initialize_surrogate_model()
        # main optimization loop
        self.step_num = 0
        if self.verbose:
            print(
                f"\n\nStarting Optimization of Task {self.task_id} with {self.optimizer} Using Seed {self.seed}...\
                Unique Run ID: {self.unique_run_id}\n"
            )
        while self.objective.num_calls < self.max_n_oracle_calls:
            # update surrogate model(s) on data
            if self.optimizer in ["vanilla", "joco", "turbo"]:
                self.update_surrogate_model()
            # generate new candidate points, evaluate them, and update data
            self.acquisition()
            # check if restart is triggered for trust reggion and restart it as needed
            self.restart_tr_as_needed()
            # update best point found so far for logging
            self.update_best()
            # log data to log dict and print update (if verbose)
            self.log_data_on_each_loop()
            self.step_num += 1

        if self.save_run_data:
            self.save_run_data_to_file()
        return self.dict_log

    def save_run_data_to_file(self):
        if not os.path.exists("run_data"):
            os.mkdir("run_data")
        filename = f"run_data/run_{self.optimizer}_{self.task_id}_seed{self.seed}_{self.unique_run_id}.json"
        with open(filename, "w") as fp:
            json.dump(self.dict_log, fp)
        print(f"\n\nRun data saved to {filename}")

    def create_one_surrgoate_model(
        self, init_data, use_dkl=True, hidden_dims=None, feature_extractor=None
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood = likelihood.to(device)
        if use_dkl:
            model = GPModelDKL(
                init_data,
                likelihood=likelihood,
                hidden_dims=hidden_dims,
                feature_extractor=feature_extractor,
            )
        else:
            model = GPModel(
                init_data,
                likelihood=likelihood,
            )
        model = model.to(device)
        mll = PredictiveLogLikelihood(
            model.likelihood, model, num_data=self.train_x.size(-2)
        )
        model = model.eval()
        return model, mll

    def initialize_surrogate_model(self):
        n_pts = min(self.train_x.shape[0], 1024)
        if self.optimizer == "joco":
            # assume g(y) model is first model in list
            self.models_list = []
            self.mlls_list = []
            # create g(y) model
            if self.use_custom_y_compression_model:
                feature_extractor = self.objective.custom_y_compression_model
            else:
                feature_extractor = None
            model, mll = self.create_one_surrgoate_model(
                init_data=self.train_y[:n_pts, :].to(device),
                hidden_dims=self.gp_y_model_hidden_dims,
                feature_extractor=feature_extractor,
                use_dkl=True,
            )
            self.models_list.append(model)
            self.mlls_list.append(mll)
            num_x_models = self.gp_y_model_hidden_dims[-1]
            # create models for each dim in final hidden layer of g(y) model
            # define shared deep kernel for gp models
            shared_feature_extractor = DenseNetwork(
                input_dim=self.train_x.size(-1), hidden_dims=self.gp_hidden_dims
            )
            shared_feature_extractor = shared_feature_extractor.to(device)
            # Define one GP per task:
            for _ in range(num_x_models):
                if self.use_dkl:
                    model = GPModelSharedDKL(
                        self.train_x[:n_pts, :].to(device),
                        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                        shared_feature_extractor=shared_feature_extractor,
                    ).to(device)
                    mll = PredictiveLogLikelihood(
                        model.likelihood, model, num_data=n_pts
                    )
                else:
                    model, mll = self.create_one_surrgoate_model(
                        init_data=self.train_x[:n_pts, :].to(device),
                        hidden_dims=self.gp_hidden_dims,
                        use_dkl=self.use_dkl,
                    )

                self.models_list.append(model)
                self.mlls_list.append(mll)
            self.model = None
            self.mll = None
        elif self.optimizer in ["random", "cma-es"]:
            # Random sampling requires no modelling
            self.models_list = None
            self.mlls_list = None
            self.model = None
            self.mll = None
        else:
            self.models_list = None
            self.mlls_list = None
            self.model, self.mll = self.create_one_surrgoate_model(
                init_data=self.train_x[:n_pts, :].to(device),
                hidden_dims=self.gp_hidden_dims,
                use_dkl=self.use_dkl,
            )

    def update_surrogate_model(self):
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            X = self.train_x
            Y = self.train_y
            S = self.train_scores.squeeze(-1)
        else:
            # otherwise, only train on most recent batch of data
            lookback = min(self.max_lookback, len(self.train_x))
            lookback = max(lookback, self.bsz)
            n_epochs = self.num_update_epochs
            X = self.train_x[-lookback:]
            Y = self.train_y[-lookback:]
            S = self.train_scores[-lookback:].squeeze(-1)
        if self.optimizer == "joco":
            self.models_list = update_surrogate_models_joco(
                models_list=self.models_list,
                mlls_list=self.mlls_list,
                learning_rte=self.learning_rte,
                train_x=X,
                train_y=Y,
                train_s=S,
                n_epochs=n_epochs,
                seed=self.seed,
                train_bsz=self.train_bsz,
                update_jointly=self.update_jointly,
                use_rand_proj_instead=self.rand_proj_baseline,
            )
        else:
            self.model = update_model(
                model=self.model,
                mll=self.mll,
                learning_rte=self.learning_rte,
                train_x=X,
                train_y=S,
                n_epochs=n_epochs,
                train_bsz=self.train_bsz,
            )

        self.initial_model_training_complete = True

    def restart_tr_as_needed(self):
        if self.tr_state.restart_triggered:
            self.tr_state = TurboState(
                dim=self.objective.input_dim,
                batch_size=self.bsz,
            )

    def acquisition(self):
        """Generate new candidate points,
        evaluate them, and update data
        """
        if (self.objective.lb is None) or (self.objective.ub is None):  # if no bounds
            absolute_bounds = None
        else:
            absolute_bounds = (self.objective.lb, self.objective.ub)
        if self.optimizer == "joco":
            x_next = generate_batch_joco(
                state=self.tr_state,
                models_list=self.models_list,
                objective=self.objective,
                X=self.train_x,
                Y=self.train_y,
                S=self.train_scores,
                batch_size=self.bsz,
                acqf=self.acq_func,
                absolute_bounds=absolute_bounds,
                use_turbo=self.use_tr,
                device=device,
                propegate_uncertainty_x=self.propegate_uncertainty_x,
                propegate_uncertainty_y=self.propegate_uncertainty_y,
                rand_proj_baseline=self.rand_proj_baseline,
            )
        elif self.optimizer == "random":
            x_next = self.get_random_x_points(n_points=self.bsz)
        elif self.optimizer == "cma-es":
            x_next_np = self.es.ask()
            x_next = torch.tensor(x_next_np, device=device, dtype=self.train_x.dtype)
        else:
            x_next = generate_batch(
                state=self.tr_state,
                model=self.model,  # GP model
                X=self.train_x,  # Evaluated points on the domain [0, 1]^d
                Y=self.train_scores,  # Function values
                batch_size=self.bsz,
                acqf=self.acq_func,  # "ei" or "ts"
                absolute_bounds=absolute_bounds,
                use_turbo=self.use_tr,
                device=device,
            )
        if len(x_next.shape) == 1:
            x_next = x_next.unsqueeze(0)
        y_next = self.objective.xs_to_ys(x_next)
        s_next = self.objective.ys_to_scores(y_next)
        self.tr_state = update_state_unconstrained(self.tr_state, s_next)
        self.train_x = torch.cat((self.train_x, x_next.detach().cpu()), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next.detach().cpu()), dim=-2)
        self.train_scores = torch.cat(
            (self.train_scores, s_next.detach().cpu()), dim=-2
        )
        if self.optimizer == "cma-es":
            s_next_np = s_next.double().detach().cpu().numpy()
            self.es.tell(x_next_np, s_next_np)
