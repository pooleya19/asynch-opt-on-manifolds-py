import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({
    "mathtext.fontset" : "cm", 
    "axes.labelsize": 14,    # axis labels
    "axes.titlesize": 14,    # subplot titles
    "xtick.labelsize": 12,   # tick numbers
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
from matplotlib.lines import Line2D
from pathlib import Path
import json


def compress_results(varied_parameters, save_directory_path):
    unique_parameter_values = [[] for _ in range(len(varied_parameters))]
    results_dict = {}
    for object in Path(save_directory_path).iterdir():
        if object.is_file():
            with open(object, "r") as file:
                data = json.load(file)
                varied_parameter_values = []
                for i in range(len(varied_parameters)):
                    varied_parameter_value = data["parameters"][varied_parameters[i]]
                    varied_parameter_values.append(varied_parameter_value)
                    if varied_parameter_value not in unique_parameter_values[i]:
                        unique_parameter_values[i].append(varied_parameter_value)
                results_dict[tuple(varied_parameter_values)] = data

    for parameter_value in unique_parameter_values:
        parameter_value.sort()

    return unique_parameter_values, results_dict


def plotExperiment(figsize, data_directory_path, figure_file_name, varied_parameter_name, varied_parameter_label, save_figure):
    # Compress all results and identify unique parameter values
    varied_parameters = ["rng_seed", varied_parameter_name]
    unique_parameter_values, results_dict = compress_results(varied_parameters, data_directory_path)

    # Parse compressed results
    rng_seed_values = unique_parameter_values[0]
    varied_parameter_values = unique_parameter_values[1]
    num_rng_seed_values = len(rng_seed_values)
    num_varied_parameter_values = len(varied_parameter_values)

    # Plot
    colors = mpl.colormaps["rainbow"](np.linspace(0, 1, num_varied_parameter_values))[:,0:3] # jet, viridis, rainbow
    fig, (ax) = plt.subplots(1,1, figsize=figsize)
    ax.set(xlabel="Iteration,  $k$", ylabel=r"Suboptimality,  $f(x(k)) - f^\star$", yscale="log")

    # Get iterations
    iterations = np.array(results_dict[(rng_seed_values[0], varied_parameter_values[0])]["iterations"])
    num_iterations = np.size(iterations)

    # Parse through all runs
    for varied_parameter_index in range(num_varied_parameter_values):
        varied_parameter_value = varied_parameter_values[varied_parameter_index]
        avg_suboptimality = np.zeros((num_iterations))
        avg_suboptimality_synch = np.zeros((num_iterations))

        for rng_seed in rng_seed_values:
            # Check run iterations
            iters = np.array(results_dict[(rng_seed,varied_parameter_value)]["iterations"])
            assert np.array_equal(iterations, iters)

            # Parse run metrics
            f = np.array(results_dict[(rng_seed, varied_parameter_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, varied_parameter_value)]["f_synch"])
            f_star = results_dict[((rng_seed, varied_parameter_value))]["f_star"]

            suboptimality = f - f_star
            suboptimality_synch = f_synch - f_star

            avg_suboptimality += suboptimality
            avg_suboptimality_synch += suboptimality_synch

            ax.plot(iters, suboptimality,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=1, alpha=0.1)
            ax.plot(iters, suboptimality_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=1, alpha=0.1)

        avg_suboptimality /= num_rng_seed_values
        avg_suboptimality_synch /= num_rng_seed_values

        ax.plot(iterations, avg_suboptimality,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=2, alpha=1)
        ax.plot(iterations, avg_suboptimality_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=2, alpha=1)

    # Generate legends
    color_handles = [Line2D([], [], color=colors[i,:], linestyle="-", linewidth=2) for i in range(num_varied_parameter_values)]
    color_labels = ["%s=%g"%(varied_parameter_label,varied_parameter_values[i]) for i in range(num_varied_parameter_values)]
    style_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2),
                      Line2D([], [], color="black", linestyle=":", linewidth=2)]
    style_labels = ["Algorithm 1", "RGD"]

    leg1 = ax.legend(color_handles, color_labels, loc="lower left")
    ax.add_artist(leg1)

    fig.canvas.draw()
    bb = leg1.get_window_extent(fig.canvas.get_renderer())
    bb = bb.transformed(ax.transAxes.inverted())

    ax.legend(style_handles, style_labels, loc="lower left", bbox_to_anchor=(bb.x1, 0), bbox_transform=ax.transAxes)

    # Save figure
    if save_figure:
        print("Saving %s... " % figure_file_name, end='', flush=True)
        fig.savefig(figure_file_name, format="pdf", bbox_inches="tight")
        print("Done!")


def main():
    figsize = (8,5)
    save_fig = True
    data_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("formation_control")
    figure_directory_path = Path(__file__).parents[3].joinpath("figures").joinpath("formation_control")
    figure_directory_path.mkdir(parents=True, exist_ok=True)

    plotExperiment(figsize, 
                   str(data_directory_path.joinpath("exp1").resolve()), 
                   str(figure_directory_path.joinpath("f_vs_B.pdf").resolve()), 
                   "B_comm", "B", save_fig)

    plotExperiment(figsize, 
                   str(data_directory_path.joinpath("exp2").resolve()), 
                   str(figure_directory_path.joinpath("f_vs_N.pdf").resolve()), 
                   "N", "N", save_fig)

    plt.show(block=True)


if __name__ == "__main__":
    main()