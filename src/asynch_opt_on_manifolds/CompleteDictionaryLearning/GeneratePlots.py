import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import json


def compress_results(varied_parameters, save_directory_path):
    unique_parameter_values = [[] for _ in range(len(varied_parameters))]
    results_dict = {}
    for object in save_directory_path.iterdir():
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


def plotExperiment(figsize, data_directory_path, figure_file_path, varied_parameter_name, varied_parameter_label, save_figure):
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
    ax.set(xlabel="Iterations", ylabel="Objective Value", yscale="log")

    for varied_parameter_index in range(num_varied_parameter_values):
        varied_parameter_value = varied_parameter_values[varied_parameter_index]
        iterations = np.array(results_dict[(rng_seed_values[0], varied_parameter_value)]["iterations"])
        num_iterations = np.size(iterations)
        avg_f = np.zeros((num_iterations))
        avg_f_synch = np.zeros((num_iterations))
        avg_sparsity = np.zeros((num_iterations))
        avg_sparsity_synch = np.zeros((num_iterations))

        for rng_seed in rng_seed_values:
            iter = np.array(results_dict[(rng_seed,varied_parameter_value)]["iterations"])
            f = np.array(results_dict[(rng_seed, varied_parameter_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, varied_parameter_value)]["f_synch"])
            sparsity = np.array(results_dict[(rng_seed, varied_parameter_value)]["sparsity"])
            sparsity_synch = np.array(results_dict[(rng_seed, varied_parameter_value)]["sparsity_synch"])

            avg_f += np.interp(iterations, iter, f)
            avg_f_synch += np.interp(iterations, iter, f_synch)
            avg_sparsity += np.interp(iterations, iter, sparsity)
            avg_sparsity_synch += np.interp(iterations, iter, sparsity_synch)

            ax.plot(iter, f,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=1, alpha=0.1)
            ax.plot(iter, f_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=1, alpha=0.1)

        avg_f /= num_rng_seed_values
        avg_f_synch /= num_rng_seed_values
        avg_sparsity /= num_rng_seed_values
        avg_sparsity_synch /= num_rng_seed_values

        ax.plot(iterations, avg_f,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=2, alpha=1)
        ax.plot(iterations, avg_f_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=2, alpha=1)

    # Generate legends
    color_handles = [Line2D([], [], color=colors[i,:], linestyle="-", linewidth=2) for i in range(num_varied_parameter_values)]
    color_labels = ["%s=%g"%(varied_parameter_label,varied_parameter_values[i]) for i in range(num_varied_parameter_values)]
    style_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2),
                      Line2D([], [], color="black", linestyle=":", linewidth=2)]
    style_labels = ["Asynch", "Synch"]

    leg1 = ax.legend(color_handles, color_labels, loc="lower left")
    ax.add_artist(leg1)

    fig.canvas.draw()
    bb = leg1.get_window_extent(fig.canvas.get_renderer())
    bb = bb.transformed(ax.transAxes.inverted())

    ax.legend(style_handles, style_labels, loc="lower left", bbox_to_anchor=(bb.x1, 0), bbox_transform=ax.transAxes)

    # Save figure
    if save_figure:
        figure_file_path_str = str(figure_file_path.resolve())
        print("Saving %s... " % figure_file_path_str, end='', flush=True)
        fig.savefig(figure_file_path_str, format="pdf", bbox_inches="tight")
        print("Done!")


def main():
    figsize = (8,5)
    data_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("dictionary_results")
    figure_directory_path = Path(__file__).parents[3].joinpath("figures").joinpath("DictionaryLearning")
    figure_directory_path.mkdir(parents=True, exist_ok=True)

    plotExperiment(figsize, data_directory_path.joinpath("exp1"), figure_directory_path.joinpath("f_vs_B.pdf"),     "B_comm",      "B", True)
    plotExperiment(figsize, data_directory_path.joinpath("exp2"), figure_directory_path.joinpath("f_vs_N.pdf"),          "N",      "N", True)
    plotExperiment(figsize, data_directory_path.joinpath("exp3"), figure_directory_path.joinpath("f_vs_Bcomm.pdf"), "B_comm", "B_comm", True)

    plt.show(block=True)


if __name__ == "__main__":
    main()