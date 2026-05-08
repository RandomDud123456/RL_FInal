import matplotlib.pyplot as plt
import numpy as np
import hydra
import os
import pandas as pd
from dopl.utils import load_results

# Set larger font sizes for the plot
plt.rcParams.update({"axes.titlesize": 36})
plt.rcParams.update({"axes.labelsize": 36})
plt.rcParams.update({"xtick.labelsize": 32})
plt.rcParams.update({"ytick.labelsize": 32})
plt.rcParams.update({"legend.fontsize": 38})
plt.rcParams.update({"figure.titlesize": 52})


def plot_training_performance(cfg, output_dir=None):
    window_size = cfg["plotting"]["window_size"]
    colors = cfg["plotting"]["algo_colors"]
    algos = cfg["algos"]
    color_map = {algo: color for algo, color in zip(algos, colors)}
    results_dicts = load_results(output_dir)
    has_oracle = all("oracle" in results_dict for results_dict in results_dicts)
    for results_dict in results_dicts:
        if has_oracle:
            for algo_name, results in results_dict.items():
                if algo_name == "oracle":
                    continue
                results["regret"] = np.array(
                    results_dict["oracle"]["reward"]
                ) - np.array(results["reward"])
                results["regret"] = results["regret"].cumsum()

    # Plot reward curves
    if has_oracle:
        fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
        axes = [ax]

    for algo_name in results_dicts[0].keys():
        rewards = [
            list(results_dict[algo_name]["reward"]) for results_dict in results_dicts
        ]
        mean_reward = np.mean(rewards, axis=0)
        std_reward = np.std(rewards, axis=0)
        # Apply moving average for smoothing
        mean_curve = (
            pd.Series(mean_reward).rolling(window=window_size, min_periods=1).mean()
        )
        std_curve = (
            pd.Series(std_reward).rolling(window=window_size, min_periods=1).mean()
        )

        # Convert to float explicitly and to numpy arrays
        mean_curve = np.array(mean_curve.astype(float))
        std_curve = np.array(std_curve.astype(float))

        x_values = range(len(mean_curve))
        axes[0].plot(
            x_values,
            mean_curve,
            label=f"{algo_name}".upper(),
            color=color_map[algo_name],
        )
        axes[0].fill_between(
            x_values,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color_map[algo_name],
        )

    axes[0].set_xlabel("Episodes")
    axes[0].set_ylabel("Episodic Reward")
    axes[0].set_title("Reward", fontsize=50)

    # Plot regret curves
    # fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    if has_oracle:
        for algo_name in results_dicts[0].keys() - {"oracle"}:
            regrets = [
                list(results_dict[algo_name]["regret"]) for results_dict in results_dicts
            ]
            mean_regret = np.mean(regrets, axis=0)
            std_regret = np.std(regrets, axis=0)
            # Apply moving average for smoothing
            mean_curve = (
                pd.Series(mean_regret).rolling(window=window_size, min_periods=1).mean()
            )
            std_curve = (
                pd.Series(std_regret).rolling(window=window_size, min_periods=1).mean()
            )

            # Convert to float explicitly and to numpy arrays
            mean_curve = np.array(mean_curve.astype(float))
            std_curve = np.array(std_curve.astype(float))

            x_values = range(len(mean_curve))
            axes[1].plot(
                x_values,
                mean_curve,
                label=f"{algo_name}".upper(),
                color=color_map[algo_name],
            )
            axes[1].fill_between(
                x_values,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                color=color_map[algo_name],
            )

        axes[1].set_xlabel("Episodes")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].set_title("Regret", fontsize=50)

    # Adding a single legend above the plots
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.0),
        fontsize=44,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to prevent label cutoff
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(
        os.path.join(output_dir, f"reward_regret_{cfg['env_config']['arm']}.pdf"),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dopl_estimation_errors(cfg):
    window_size = cfg["plotting"]["window_size"]
    colors = cfg["plotting"]["algo_colors"]
    algos = cfg["algos"]
    color_map = {algo: color for algo, color in zip(algos, colors)}
    results_dicts = load_results()
    # Plot estimation_errors of DOPL algorithm
    fig, ax = plt.subplots(figsize=(15, 10))
    plotted_keys = set()
    for results_dict in results_dicts:
        if "dopl" in results_dict:
            dopl_results = results_dict["dopl"]
            error_keys = [key for key in dopl_results if key.endswith("_error")]

            for error_key in error_keys:
                if error_key not in plotted_keys:
                    errors = [
                        list(results_dict["dopl"][error_key])
                        for results_dict in results_dicts
                    ]
                    mean_error = np.mean(errors, axis=0)
                    std_error = np.std(errors, axis=0)
                    # Apply moving average for smoothing
                    mean_curve = (
                        pd.Series(mean_error)
                        .rolling(window=window_size, min_periods=1)
                        .mean()
                    )
                    std_curve = (
                        pd.Series(std_error)
                        .rolling(window=window_size, min_periods=1)
                        .mean()
                    )

                    # Convert to float explicitly and to numpy arrays
                    mean_curve = np.array(mean_curve.astype(float))
                    std_curve = np.array(std_curve.astype(float))

                    x_values = range(len(mean_curve))
                    ax.plot(
                        x_values,
                        mean_curve,
                        label=f"{error_key}",
                    )
                    ax.fill_between(
                        x_values,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2,
                    )
                    plotted_keys.add(error_key)
        else:
            return

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Estimation Error")

    # Adding a single legend above the plot
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.15),
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(
        os.path.join(
            output_dir, f"dopl_estimation_errors_{cfg['env_config']['arm']}.pdf"
        ),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_direct_wibql_errors(cfg):
    window_size = cfg["plotting"]["window_size"]
    colors = cfg["plotting"]["algo_colors"]
    algos = cfg["algos"]
    color_map = {algo: color for algo, color in zip(algos, colors)}
    results_dicts = load_results()
    # Plot estimation_errors of DOPL algorithm
    fig, ax = plt.subplots(figsize=(15, 10))
    plotted_keys = set()
    for results_dict in results_dicts:
        if "direct_wibql" in results_dict:
            wibql_results = results_dict["direct_wibql"]
            error_keys = [key for key in wibql_results if key.endswith("_error")]

            for error_key in error_keys:
                if error_key not in plotted_keys:
                    errors = [
                        list(results_dict["direct_wibql"][error_key])
                        for results_dict in results_dicts
                    ]
                    mean_error = np.mean(errors, axis=0)
                    std_error = np.std(errors, axis=0)
                    # Apply moving average for smoothing
                    mean_curve = (
                        pd.Series(mean_error)
                        .rolling(window=window_size, min_periods=1)
                        .mean()
                    )
                    std_curve = (
                        pd.Series(std_error)
                        .rolling(window=window_size, min_periods=1)
                        .mean()
                    )

                    # Convert to float explicitly and to numpy arrays
                    mean_curve = np.array(mean_curve.astype(float))
                    std_curve = np.array(std_curve.astype(float))

                    x_values = range(len(mean_curve))
                    ax.plot(
                        x_values,
                        mean_curve,
                        label=f"{error_key}",
                    )
                    ax.fill_between(
                        x_values,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2,
                    )
                    plotted_keys.add(error_key)
        else:
            return

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Estimation Error")

    # Adding a single legend above the plot
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        bbox_to_anchor=(0.5, -0.15),
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    plt.savefig(
        os.path.join(
            output_dir, f"direct_wibql_estimation_errors_{cfg['env_config']['arm']}.pdf"
        ),
        bbox_extra_artists=(legend,),
        bbox_inches="tight",
    )
    plt.close(fig)
