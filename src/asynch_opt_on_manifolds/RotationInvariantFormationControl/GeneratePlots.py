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


def main():
    # ========== Experiment 1 ==========
    # Define experiment features
    varied_parameters = ["rng_seed", "B_comm"]
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("exp1")

    # Compress all results and identify unique parameter values
    unique_parameter_values, results_dict = compress_results(varied_parameters, save_directory_path)

    # Plot
    num_rng_seeds = len(unique_parameter_values[0])
    num_deps = len(unique_parameter_values[1])
    # colors = mpl.colormaps["jet"](np.linspace(0.15, 0.85, num_deps))[:,0:3]
    colors = mpl.colormaps["rainbow"](np.linspace(0, 1, num_deps))[:,0:3]

    fig1, (ax1) = plt.subplots(1,1, figsize=(8,8))
    ax1.set(xlabel="Iterations", ylabel="Objective Value", yscale="log")

    iterations = np.array(results_dict[(unique_parameter_values[0][0], unique_parameter_values[1][0])]["iterations"])
    num_iterations = np.size(iterations)

    for dep_index in range(num_deps):
        dep_value = unique_parameter_values[1][dep_index]
        avg_f = np.zeros((num_iterations))
        avg_f_synch = np.zeros((num_iterations))

        for rng_seed in unique_parameter_values[0]:
            f = np.array(results_dict[(rng_seed, dep_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, dep_value)]["f_synch"])

            avg_f += f
            avg_f_synch += f_synch

            ax1.plot(iterations, f, color=colors[dep_index,:], linestyle="-", linewidth=1, alpha=0.1)
            ax1.plot(iterations, f_synch, color=colors[dep_index,:], linestyle=":", linewidth=1, alpha=0.1)

        avg_f /= num_rng_seeds
        avg_f_synch /= num_rng_seeds

        ax1.plot(iterations, avg_f, color=colors[dep_index,:], linestyle="-", linewidth=2, alpha=1)
        ax1.plot(iterations, avg_f_synch, color=colors[dep_index,:], linestyle=":", linewidth=2, alpha=1)

    # Generate legends
    color_handles = [Line2D([], [], color=colors[i,:], linestyle="-", linewidth=2) for i in range(num_deps)]
    color_labels = ["B=%g"%unique_parameter_values[1][i]  for i in range(num_deps)]
    style_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2),
                      Line2D([], [], color="black", linestyle=":", linewidth=2)]
    style_labels = ["Asynch", "Synch"]

    leg1 = ax1.legend(color_handles, color_labels, loc="upper right")
    ax1.add_artist(leg1)

    fig1.canvas.draw()
    bb = leg1.get_window_extent(fig1.canvas.get_renderer())
    bb = bb.transformed(ax1.transAxes.inverted())

    ax1.legend(style_handles, style_labels, loc="upper right", bbox_to_anchor=(bb.x0, 1), bbox_transform=ax1.transAxes)


    plt.show(block=True)





if __name__ == "__main__":
    main()