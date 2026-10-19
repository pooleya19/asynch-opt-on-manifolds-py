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


def plotExperiment1(figsize, figure_directory_path, figure_file_name):
    # Define experiment features
    varied_parameters = ["rng_seed", "B_comm"]
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("exp1")

    # Compress all results and identify unique parameter values
    unique_parameter_values, results_dict = compress_results(varied_parameters, save_directory_path)

    # Plot
    num_rng_seeds = len(unique_parameter_values[0])
    num_deps = len(unique_parameter_values[1])
    colors = mpl.colormaps["rainbow"](np.linspace(0, 1, num_deps))[:,0:3] # jet, viridis, rainbow

    fig1, (ax1) = plt.subplots(1,1, figsize=figsize)
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

    leg1 = ax1.legend(color_handles, color_labels, loc="lower left")
    ax1.add_artist(leg1)

    fig1.canvas.draw()
    bb = leg1.get_window_extent(fig1.canvas.get_renderer())
    bb = bb.transformed(ax1.transAxes.inverted())

    ax1.legend(style_handles, style_labels, loc="lower left", bbox_to_anchor=(bb.x1, 0), bbox_transform=ax1.transAxes)

    # Save figure
    print("Saving %s... " % figure_file_name, end='', flush=True)
    figure_file_path = str(figure_directory_path.joinpath(figure_file_name).resolve())
    fig1.savefig(figure_file_path, format="svg", bbox_inches="tight")
    print("Done!")


def plotExperiment2(figsize, figure_directory_path, figure_file_name):
    # Define experiment features
    varied_parameters = ["rng_seed", "N"]
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("exp2")

    # Compress all results and identify unique parameter values
    unique_parameter_values, results_dict = compress_results(varied_parameters, save_directory_path)

    # Plot
    num_rng_seeds = len(unique_parameter_values[0])
    num_deps = len(unique_parameter_values[1])
    colors = mpl.colormaps["rainbow"](np.linspace(0, 1, num_deps))[:,0:3] # jet, viridis, rainbow

    fig2, (ax2) = plt.subplots(1,1, figsize=figsize)
    ax2.set(xlabel="Iterations", ylabel="Objective Value / N", yscale="log")

    iterations = np.array(results_dict[(unique_parameter_values[0][0], unique_parameter_values[1][0])]["iterations"])
    num_iterations = np.size(iterations)

    for dep_index in range(num_deps):
        dep_value = unique_parameter_values[1][dep_index]
        avg_f = np.zeros((num_iterations))
        avg_f_synch = np.zeros((num_iterations))

        for rng_seed in unique_parameter_values[0]:
            f = np.array(results_dict[(rng_seed, dep_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, dep_value)]["f_synch"])

            f /= dep_value
            f_synch /= dep_value

            avg_f += f
            avg_f_synch += f_synch

            ax2.plot(iterations, f, color=colors[dep_index,:], linestyle="-", linewidth=1, alpha=0.1)
            ax2.plot(iterations, f_synch, color=colors[dep_index,:], linestyle=":", linewidth=1, alpha=0.1)

        avg_f /= num_rng_seeds
        avg_f_synch /= num_rng_seeds

        ax2.plot(iterations, avg_f, color=colors[dep_index,:], linestyle="-", linewidth=2, alpha=1)
        ax2.plot(iterations, avg_f_synch, color=colors[dep_index,:], linestyle=":", linewidth=2, alpha=1)

    # Generate legends
    color_handles = [Line2D([], [], color=colors[i,:], linestyle="-", linewidth=2) for i in range(num_deps)]
    color_labels = ["N=%g"%unique_parameter_values[1][i]  for i in range(num_deps)]
    style_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2),
                      Line2D([], [], color="black", linestyle=":", linewidth=2)]
    style_labels = ["Asynch", "Synch"]

    leg1 = ax2.legend(color_handles, color_labels, loc="lower left")
    ax2.add_artist(leg1)

    fig2.canvas.draw()
    bb = leg1.get_window_extent(fig2.canvas.get_renderer())
    bb = bb.transformed(ax2.transAxes.inverted())

    ax2.legend(style_handles, style_labels, loc="lower left", bbox_to_anchor=(bb.x1, 0), bbox_transform=ax2.transAxes)

    # Save figure
    print("Saving %s... " % figure_file_name, end='', flush=True)
    figure_file_path = str(figure_directory_path.joinpath(figure_file_name).resolve())
    fig2.savefig(figure_file_path, format="svg", bbox_inches="tight")
    print("Done!")


def plotExperiment3(figsize, figure_directory_path, figure_file_name):
    # Define experiment features
    varied_parameters = ["rng_seed", "B_comp"]
    save_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("exp3")

    # Compress all results and identify unique parameter values
    unique_parameter_values, results_dict = compress_results(varied_parameters, save_directory_path)

    # Plot
    num_rng_seeds = len(unique_parameter_values[0])
    num_deps = len(unique_parameter_values[1])
    colors = mpl.colormaps["rainbow"](np.linspace(0, 1, num_deps))[:,0:3] # jet, viridis, rainbow

    fig3, (ax3) = plt.subplots(1,1, figsize=figsize)
    ax3.set(xlabel="Iterations", ylabel="Objective Value", yscale="log")

    iterations = np.array(results_dict[(unique_parameter_values[0][0], unique_parameter_values[1][0])]["iterations"])
    num_iterations = np.size(iterations)

    for dep_index in range(num_deps):
        dep_value = unique_parameter_values[1][dep_index]
        if dep_value == 1:
            continue
        avg_f = np.zeros((num_iterations))
        avg_f_synch = np.zeros((num_iterations))

        for rng_seed in unique_parameter_values[0]:
            f = np.array(results_dict[(rng_seed, dep_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, dep_value)]["f_synch"])

            avg_f += f
            avg_f_synch += f_synch

            ax3.plot(iterations, f, color=colors[dep_index,:], linestyle="-", linewidth=1, alpha=0.1)
            ax3.plot(iterations, f_synch, color=colors[dep_index,:], linestyle=":", linewidth=1, alpha=0.1)

        avg_f /= num_rng_seeds
        avg_f_synch /= num_rng_seeds

        ax3.plot(iterations, avg_f, color=colors[dep_index,:], linestyle="-", linewidth=2, alpha=1)
        ax3.plot(iterations, avg_f_synch, color=colors[dep_index,:], linestyle=":", linewidth=2, alpha=1)

    # Generate legends
    color_handles = [Line2D([], [], color=colors[i,:], linestyle="-", linewidth=2) for i in range(num_deps)]
    color_labels = ["B_comp=%g"%unique_parameter_values[1][i]  for i in range(num_deps)]
    style_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2),
                      Line2D([], [], color="black", linestyle=":", linewidth=2)]
    style_labels = ["Asynch", "Synch"]

    leg1 = ax3.legend(color_handles, color_labels, loc="lower left")
    ax3.add_artist(leg1)

    fig3.canvas.draw()
    bb = leg1.get_window_extent(fig3.canvas.get_renderer())
    bb = bb.transformed(ax3.transAxes.inverted())

    ax3.legend(style_handles, style_labels, loc="lower left", bbox_to_anchor=(bb.x1, 0), bbox_transform=ax3.transAxes)

    # Save figure
    print("Saving %s... " % figure_file_name, end='', flush=True)
    figure_file_path = str(figure_directory_path.joinpath(figure_file_name).resolve())
    fig3.savefig(figure_file_path, format="svg", bbox_inches="tight")
    print("Done!")


def main():
    figure_directory_path = Path(__file__).parents[3].joinpath("figures").joinpath("RotationInvariantFormationControl")
    figure_directory_path.mkdir(parents=True, exist_ok=True)
    figsize = (8,5)

    plotExperiment1(figsize, figure_directory_path, "f_vs_B.svg")
    plotExperiment2(figsize, figure_directory_path, "f_vs_N.svg")
    plotExperiment3(figsize, figure_directory_path, "f_vs_Bcomp.svg")
    plt.show(block=True)


if __name__ == "__main__":
    main()