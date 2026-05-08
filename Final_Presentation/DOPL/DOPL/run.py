import hydra
from omegaconf import DictConfig

from dopl.utils import (
    load_environment_configuration,
    initialize_environment,
)
from dopl.train import get_opt_performance, train
from dopl.plot import (
    plot_training_performance,
    plot_dopl_estimation_errors,
    plot_direct_wibql_errors,
)
import warnings
import logging

logging.getLogger("pyomo").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="cpap", version_base=None)
def main(cfg: DictConfig):
    # instantiate env
    P_list, R_list, arm_constraint = load_environment_configuration(cfg)
    env = initialize_environment(cfg, P_list, R_list, arm_constraint)
    get_opt_performance(env)

    # Perform training
    train(cfg, env)

    # Plot results
    plot_training_performance(cfg)
    plot_dopl_estimation_errors(cfg)
    plot_direct_wibql_errors(cfg)


if __name__ == "__main__":
    main()
