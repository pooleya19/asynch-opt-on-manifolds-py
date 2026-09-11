import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

    print("Initial objective values:")
    for key in list(results_dict.keys()):
        f = results_dict[key]["f"]
        f_synch = results_dict[key]["f_synch"]
        print("\t%9s: f=%.2f, f_synch=%.2f" % (str(key),f[0],f_synch[0]))

    # Plot
    num_rng_seeds = len(unique_parameter_values[0])
    num_deps = len(unique_parameter_values[1])
    colors = mpl.colormaps["jet"](np.linspace(0.15, 0.85, num_deps))[:,0:3]

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

        ax1.plot(iterations, avg_f, color=colors[dep_index,:], linestyle="-", linewidth=2, alpha=1, label="Asynch B=%g"%dep_value)
        ax1.plot(iterations, avg_f_synch, color=colors[dep_index,:], linestyle=":", linewidth=2, alpha=1, label="Synch B=%g"%dep_value)

    ax1.legend(loc="upper right")
    
    plt.show(block=True)





if __name__ == "__main__":
    main()