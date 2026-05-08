from dopl.registry import TRAINING_FUNCTIONS
import importlib
import logging
import wandb
from dopl.utils import set_seed, save_results

log = logging.getLogger(__name__)


def train(cfg, env):
    """Train and evaluate the model based on the given configuration."""
    log.info(f"Running experiment {cfg.exp.name}")
    for seed in range(cfg.num_seeds):
        print_string = ("*" * 10, f"Seed {seed + 1}", "*" * 10)
        log.info(" ".join(print_string))
        results_dict = train_algos(env, cfg, seed)
        save_results(results_dict, seed)  # Save results dict in hydra output


def train_algos(env, cfg, seed):
    """Run each algo and aggregate results"""
    results_dict = {}
    for module_name in ("dopl.train.algos.dopl", "dopl.train.algos.mle_lp"):
        importlib.import_module(module_name)

    for algo_name in cfg.algos:
        set_seed(seed)
        config_dict = dict(cfg)
        config_dict["algo"] = algo_name
        wandb.init(
            project="dopl",
            name=f"{cfg.exp.name}",
            job_type=f"{algo_name}_seed_{seed + 1}",
            config=config_dict,
            mode=cfg.get("wandb_mode", "disabled"),
            reinit=False,
        )
        log.info(f"Training {algo_name}")
        results = TRAINING_FUNCTIONS[algo_name](env, cfg)
        results_dict[algo_name] = results
        log.info(f"Run time for {algo_name}: {results['run_time']:.2f} seconds")
        wandb.finish()
    return results_dict
