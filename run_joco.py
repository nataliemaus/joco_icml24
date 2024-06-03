import argparse

from joco.run_optimization import Optimize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_id",
        help="id string for optimization task to run, see README.md for details",
        type=str,
        choices=[
            "langermann",
            "rosenbrock",
            "rover",
            "env",
            "pde",
            "sportscar",
            "dog",
            "aircraft",
            "falcon",
        ],
        default="langermann",
        required=False,
    )
    parser.add_argument(
        "--seed",
        help="Random seed to be set.",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--max_n_oracle_calls",
        help="Max number of oracle calls before optimization terminates (budget). If -1, use default for task",
        type=int,
        default=-1,
        required=False,
    )
    parser.add_argument(
        "--learning_rte",
        help="Learning rate for model updates with Adam optimizer",
        type=float,
        default=0.01,
        required=False,
    )
    parser.add_argument(
        "--bsz",
        help="Acquisition batch size",
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        "--acq_func",
        help=" string indiciating which acquisition function to use (Expected Improvement or Thompson Sampling)",
        type=str,
        default="ts",
        required=False,
        choices=[
            "ts",
            "ei",
        ],
    )
    parser.add_argument(
        "--num_initialization_points",
        help="Number evaluated data points used to optimization initialize run. If -1, use default for task",
        type=int,
        default=-1,
        required=False,
    )
    parser.add_argument(
        "--init_n_epochs",
        help="Number of epochs to train the surrogate model for on initial data before optimization begins",
        type=int,
        default=30,
        required=False,
    )
    parser.add_argument(
        "--input_dim",
        help="Input search space dimension of optimizatoin task. If None provided, we use default for task",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--output_dim",
        help="Intermediate output dimension of optimizatoin task. If None provided, we use default for task",
        type=int,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_update_epochs",
        help="Number of epochs to update the model(s) for on each optimization step",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--optimizer",
        help="string id indicates which optimizer to use (joco, turbo, or vanilla for Vanilla BO)",
        type=str,
        choices=["joco", "turbo", "vanilla", "random", "cma-es"],
        default="joco",
        required=False,
    )
    parser.add_argument(
        "--update_jointly",
        help=" if True, update models jointly when running JoCo",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--propegate_uncertainty_x",
        help=" if True, propegate uncertainty through gpx models during ts when running JoCo",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--propegate_uncertainty_y",
        help=" if True, propegate uncertainty through gpy models during ts when running JoCo",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--train_bsz",
        help=" batch size upsed for updating surrogate model(s) on data",
        type=int,
        default=64,
        required=False,
    )
    parser.add_argument(
        "--verbose",
        help=" if True, print progress update on each iteration of BO",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--save_run_data",
        help=" if True, save all data from optimization run locally after run finishes",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--use_tr",
        help=" if True, use trust regions for JoCo (only relevant when running JoCo)",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--use_dkl",
        help=" if True, use GP with a deep kernel to compress the input space",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--rand_proj_baseline",
        help=" if True, run joco w/ random proj baseline",
        type=str2bool,
        default=False,
        required=False,
    )
    args = parser.parse_args()
    runner = Optimize(
        task_id=args.task_id,
        seed=args.seed,
        max_n_oracle_calls=args.max_n_oracle_calls,
        learning_rte=args.learning_rte,
        bsz=args.bsz,
        num_initialization_points=args.num_initialization_points,
        init_n_epochs=args.init_n_epochs,
        num_update_epochs=args.num_update_epochs,
        optimizer=args.optimizer,
        verbose=args.verbose,
        train_bsz=args.train_bsz,
        use_dkl=args.use_dkl,
        use_tr=args.use_tr,
        save_run_data=args.save_run_data,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        acq_func=args.acq_func,
        update_jointly=args.update_jointly,
        propegate_uncertainty_x=args.propegate_uncertainty_x,
        propegate_uncertainty_y=args.propegate_uncertainty_y,
        rand_proj_baseline=args.rand_proj_baseline,
    )
    run_data_dict = runner.run_optimization()

