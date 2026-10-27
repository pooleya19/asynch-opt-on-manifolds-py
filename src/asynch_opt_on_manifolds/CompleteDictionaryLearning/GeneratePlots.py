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

    # Init suboptimality reductions
    suboptimality_reductions = []

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

        suboptimality_reductions.append(avg_suboptimality_synch[-1] / avg_suboptimality[-1])

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

    suboptimality_reductions = np.floor(suboptimality_reductions)

    print("Vary %s: Suboptimality reduction: %.1f-%.1f" % (varied_parameter_label,np.min(suboptimality_reductions), np.max(suboptimality_reductions)))
    for varied_parameter_index in range(num_varied_parameter_values):
        varied_parameter_value = varied_parameter_values[varied_parameter_index]
        print("\t%s=%g: %g" % (varied_parameter_label, varied_parameter_value, suboptimality_reductions[varied_parameter_index]))

    # Save figure
    if save_figure:
        print("Saving %s... " % figure_file_name, end='', flush=True)
        fig.savefig(figure_file_name, format="pdf", bbox_inches="tight")
        print("Done!")


def plotVisuals(option, figure_data_file_name, figure_file_name, save_figure):
    # Define parameters
    if option==1:
        figsize = (6,5)
        m = 5
        recon_indices = list(range(10))
        recon_rows = 5
        recon_cols = int(np.ceil(len(recon_indices)/recon_rows))
        subfig_rows = 1
        subfig_cols = 3
        name_original = "Dataset\n" + r"($Y$)"
        name_asynch = "Reconstruction\n(Algorithm 1)"
        name_synch = "Reconstruction\n(RGD)"
    elif option==2:
        figsize = (5,2)
        m = 5
        recon_indices = [0,1,3,9]
        recon_rows = 2
        recon_cols = int(np.ceil(len(recon_indices)/recon_rows))
        subfig_rows = 1
        subfig_cols = 3
        name_original = "Dataset\n" + r"($Y$)"
        name_asynch = "Reconstruction\n(Algorithm 1)"
        name_synch = "Reconstruction\n(RGD)"
    elif option==3:
        figsize = (6,3)
        m = 5
        recon_indices = list(range(9))
        recon_rows = 1
        recon_cols = int(np.ceil(len(recon_indices)/recon_rows))
        subfig_rows = 3
        subfig_cols = 1
        name_original = "Dataset " + r"($Y$)"
        name_asynch = "Reconstruction (Algorithm 1)"
        name_synch = "Reconstruction (RGD)"
    
    ieee_width = 3.5
    text_fontsize = 8

    # Load data
    figure_data = None
    with open(figure_data_file_name, "r") as file:
        figure_data = json.load(file)
    if figure_data is None:
        raise RuntimeError("Failed to load data.")

    # Parse data
    Y = np.array(figure_data["Y"])
    Ypp = np.array(figure_data["Ypp"])
    Y_mean = np.array(figure_data["Y_mean"])
    img_shape = figure_data["img_shape"]
    Zinv = np.array(figure_data["Zinv"])
    A = np.array(figure_data["A"])
    A_synch = np.array(figure_data["A_synch"])

    # Generate plot
    v_min_Y = np.min(np.abs(Y))
    v_max_Y = np.max(np.abs(Y))
    fig2 = plt.figure(figsize=figsize, layout="constrained")
    subfigs2 = fig2.subfigures(subfig_rows, subfig_cols, wspace=0.1, hspace=0.1)
    recon_images = {}
    recon_names = [name_original, name_asynch, name_synch]
    for subfig, name in zip(subfigs2, recon_names):
        subfig.suptitle(name, fontsize=text_fontsize/ieee_width*figsize[0])
        # subfig.supylabel(name)
        axes = subfig.subplots(recon_rows, recon_cols)
        handles = []
        for k, ax in enumerate(axes.flat):
            ax.set_axis_off()
            if k < len(recon_indices):
                handles.append(ax.imshow(np.zeros(img_shape), cmap="gray", interpolation="nearest", clim=(v_min_Y,v_max_Y)))
        recon_images[name] = handles

    # Original images
    for i in range(len(recon_indices)):
        # img_og = np.reshape(Y[:,[recon_indices[i]]], img_shape)
        img_og = np.reshape(Zinv @ Ypp[:,[recon_indices[i]]] + Y_mean, img_shape)
        recon_images[name_original][i].set(data=img_og)

    # Top-m reconstructions
    C = A.T @ Ypp[:,recon_indices]
    C_synch = A_synch.T @ Ypp[:,recon_indices]

    # Top-m reconstruction
    Cm = np.zeros_like(C)
    Cm_synch = np.zeros_like(C_synch)
    for i in range(len(recon_indices)):
        index_i = np.argpartition(np.abs(C[:,i]),-m)[-m:]
        index_i_synch = np.argpartition(np.abs(C_synch[:,i]),-m)[-m:]
        Cm[index_i,i] = C[index_i,i]
        Cm_synch[index_i_synch,i] = C_synch[index_i_synch,i]

    for i in range(len(recon_indices)):
        img = np.reshape(Zinv @ A @ Cm[:,[i]] + Y_mean, img_shape)
        recon_images[name_asynch][i].set(data=img)

        img_synch = np.reshape(Zinv @ A_synch @ Cm_synch[:,[i]] + Y_mean, img_shape)
        recon_images[name_synch][i].set(data=img_synch)

    # Save figure
    if save_figure:
        print("Saving %s... " % figure_file_name, end='', flush=True)
        fig2.savefig(figure_file_name, format="pdf", bbox_inches="tight")
        print("Done!")


def main():
    figsize = (8,5)
    save_fig = True
    data_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("dictionary_learning")
    figure_directory_path = Path(__file__).parents[3].joinpath("figures").joinpath("dictionary_learning")
    figure_directory_path.mkdir(parents=True, exist_ok=True)

    plotExperiment(figsize, 
                   str(data_directory_path.joinpath("exp1").resolve()), 
                   str(figure_directory_path.joinpath("f_vs_B.pdf").resolve()), 
                   "B_comm", "B", save_fig)

    plotExperiment(figsize, 
                   str(data_directory_path.joinpath("exp2").resolve()), 
                   str(figure_directory_path.joinpath("f_vs_N.pdf").resolve()), 
                   "N", "N", save_fig)

    plotVisuals(3,
                str(data_directory_path.joinpath("figure_data.json").resolve()),
                str(figure_directory_path.joinpath("figure.pdf").resolve()),
                save_fig)

    plt.show(block=True)


if __name__ == "__main__":
    main()