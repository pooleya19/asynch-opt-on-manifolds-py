import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({
    "mathtext.fontset" : "cm",
    "pdf.fonttype" : 42,   # embed TrueType fonts instead of Type 3
    "ps.fonttype" : 42, 
    "axes.labelsize": 14,    # axis labels
    "axes.titlesize": 14,    # subplot titles
    "xtick.labelsize": 12,   # tick numbers
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.family": "serif",
})
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
import json

from asynch_opt_on_manifolds.Utils.RotationUtils import quat_to_rot_mat

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
        print("\tSaving %s... " % figure_file_name, end='', flush=True)
        fig.savefig(figure_file_name, format="pdf", bbox_inches="tight")
        print("Done!")


def generate_formation_visuals(x, formation_edges, axis_length):
    N = len(x)
    ts = [x[i][0] for i in range(N)]
    Rs = [quat_to_rot_mat(x[i][1]) for i in range(N)]

    positions = np.zeros((N,3))
    x_axes = np.zeros((N,2,3))
    y_axes = np.zeros((N,2,3))
    z_axes = np.zeros((N,2,3))
    for i in range(N):
        positions[i,:] = ts[i][:,0]
        x_axes[i,0,:] = ts[i][:,0]
        x_axes[i,1,:] = ts[i][:,0] + axis_length*Rs[i][:,0]
        y_axes[i,0,:] = ts[i][:,0]
        y_axes[i,1,:] = ts[i][:,0] + axis_length*Rs[i][:,1]
        z_axes[i,0,:] = ts[i][:,0]
        z_axes[i,1,:] = ts[i][:,0] + axis_length*Rs[i][:,2]

    edges = np.zeros((len(formation_edges),2,3))
    for k in range(len(formation_edges)):
        i,j = formation_edges[k]
        edges[k,0,:] = ts[i][:,0]
        edges[k,1,:] = ts[j][:,0]

    return positions, x_axes, y_axes, z_axes, edges

def load_x(x_list):
    return [[np.array(x_list[i][0]), np.array(x_list[i][1])] for i in range(len(x_list))]

def plotVisuals1(figure_data_file_name, figure_file_name, save_figure):
    figsize = (10,10)

    # Load data
    figure_data = None
    with open(figure_data_file_name, "r") as file:
        figure_data = json.load(file)
    if figure_data is None:
        raise RuntimeError("Failed to load data.")

    # Parse data
    x_formation = load_x(figure_data["x_formation"])
    x_initial = load_x(figure_data["x_initial"])
    x = load_x(figure_data["x"])
    x_synch = load_x(figure_data["x_synch"])
    formation_edges = figure_data["formation_edges"]

    # Generate plot
    fig2, ax_form = plt.subplots(2, 2, figsize=figsize, dpi=150, subplot_kw={"projection": "3d"})
    ((ax_form_des, ax_form_init), (ax_form_asynch, ax_form_synch)) = ax_form
    axes_length = 0.1
    axes_linewidth = 1.5
    edge_linewidth = 0.5

    lim = 0.8
    positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_formation, formation_edges, axes_length)
    ax_form_des.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3)[0]
    ax_form_des.add_collection3d(Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth))
    ax_form_des.add_collection3d(Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth))
    ax_form_des.add_collection3d(Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth))
    ax_form_des.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=edge_linewidth))
    ax_form_des.set(title="Desired Formation", xlim=(-lim,lim), ylim=(-lim,lim), zlim=(-lim,lim), aspect="equal")
    
    positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_initial, formation_edges, axes_length)
    ax_form_init.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3)[0]
    ax_form_init.add_collection3d(Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth))
    ax_form_init.add_collection3d(Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth))
    ax_form_init.add_collection3d(Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth))
    ax_form_init.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=edge_linewidth))
    ax_form_init.set(title="Initial Poses", xlim=(-lim,lim), ylim=(-lim,lim), zlim=(-lim,lim), aspect="equal")
    
    positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x, formation_edges, axes_length)
    ax_form_asynch.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3)[0]
    ax_form_asynch.add_collection3d(Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth))
    ax_form_asynch.add_collection3d(Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth))
    ax_form_asynch.add_collection3d(Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth))
    ax_form_asynch.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=edge_linewidth))
    ax_form_asynch.set(title="Final Poses (Algorithm 1)", xlim=(-lim,lim), ylim=(-lim,lim), zlim=(-lim,lim), aspect="equal")
    
    positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_synch, formation_edges, axes_length)
    ax_form_synch.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3)[0]
    ax_form_synch.add_collection3d(Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth))
    ax_form_synch.add_collection3d(Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth))
    ax_form_synch.add_collection3d(Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth))
    ax_form_synch.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=edge_linewidth))
    ax_form_synch.set(title="Final Poses (RGD)", xlim=(-lim,lim), ylim=(-lim,lim), zlim=(-lim,lim), aspect="equal")

    # Save figure
    if save_figure:
        print("Saving %s... " % figure_file_name, end='', flush=True)
        fig2.savefig(figure_file_name, format="pdf", bbox_inches="tight")
        print("Done!")

def plotVisuals2(figure_data_file_name, figure_file_name, save_figure, show_dividers):
    figsize = (10,6.6)
    ieee_width = 3.5

    text_fontsize = 8
    axes_fontsize = 5

    # Load data
    figure_data = None
    with open(figure_data_file_name, "r") as file:
        figure_data = json.load(file)
    if figure_data is None:
        raise RuntimeError("Failed to load data.")

    # Parse data
    x_formation = load_x(figure_data["x_formation"])
    x_initial = load_x(figure_data["x_initial"])
    x = load_x(figure_data["x"])
    x_synch = load_x(figure_data["x_synch"])
    formation_edges = figure_data["formation_edges"]

    # Generate plot
    fig2 = plt.figure(figsize=figsize, dpi=150)
    axes_length = 0.1
    axes_linewidth = 1.5
    edge_linewidth = 0.5

    lim_xy = 0.72
    lim_z = 0.5
    # gap = 0.03
    gap = 0
    cam_elev = 18
    cam_azim = -60
    caption_height = 0.07
    panel_height = (1-gap)/2

    divider_color = 0.8*np.ones((3))
    divider_linewidth = 0.8

    # Define panels
    # panels = [(x_formation, "(a) Desired Formation", (0.0, panel_height + gap)),
    #           (x_initial,   "(b) Initial Poses",     (0.5, panel_height + gap)),
    #           (x,           "(c) Final Poses (Algorithm 1)",       (0.0, 0.0)),
    #           (x_synch,     "(d) Final Poses (RGD)",               (0.5, 0.0))]
    panels = [(x_formation, "Desired Formation", (0.0, panel_height + gap)),
              (x_initial,   "Initial Poses",     (0.5, panel_height + gap)),
              (x,           "Final Poses (Algorithm 1)",       (0.0, 0.0)),
              (x_synch,     "Final Poses (RGD)",               (0.5, 0.0))]

    for x_plot, caption, (x0, y0) in panels:
        ax = fig2.add_axes([x0, y0+caption_height, 0.5, panel_height-caption_height], projection="3d")

        # Plot formations
        positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_plot, formation_edges, axes_length)
        ax.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3, clip_on=False)
        ax.add_collection3d(Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth, clip_on=False))
        ax.add_collection3d(Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth, clip_on=False))
        ax.add_collection3d(Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth, clip_on=False))
        ax.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=edge_linewidth, clip_on=False))
        ax.set(xlim=(-lim_xy,lim_xy), ylim=(-lim_xy,lim_xy), zlim=(-lim_z,lim_z))
        ax.set_box_aspect((1, 1, lim_z/lim_xy), zoom=1.8)
        ax.view_init(elev=cam_elev, azim=cam_azim)
        ax.axis("off")

        # Plot title
        fig2.text(x0+0.25, y0+caption_height/2, caption, ha="center", va="center", fontsize=text_fontsize/ieee_width*figsize[0])

        # Plot x-y-z axes indicator
        ax_xyz = fig2.add_axes([x0+0.015, y0+caption_height+0.02, 0.07, 0.11], projection="3d")
        for c, (name, color) in enumerate(zip("xyz", "rgb")):
            e = np.zeros(3)
            e[c] = 1
            ax_xyz.add_collection3d(Line3DCollection([[np.zeros(3), e]], colors=color, linewidths=axes_linewidth))
            ax_xyz.text(1.35*e[0], 1.35*e[1], 1.35*e[2], "$%s$" % name, fontsize=axes_fontsize/ieee_width*figsize[0], ha="center", va="center")
        ax_xyz.set(xlim=(-0.2,1.2), ylim=(-0.2,1.2), zlim=(-0.2,1.2))
        ax_xyz.set_box_aspect((1,1,1))
        ax_xyz.view_init(elev=cam_elev, azim=cam_azim)
        ax_xyz.axis("off")
            
    # Plot dividers between panels
    if show_dividers:
        fig2.add_artist(Line2D([0.02, 0.98], [0.5, 0.5], color=divider_color, linewidth=divider_linewidth))
        fig2.add_artist(Line2D([0.5, 0.5], [0.02, 0.98], color=divider_color, linewidth=divider_linewidth))

    # Save figure
    if save_figure:
        print("Saving %s... " % figure_file_name, end='', flush=True)
        fig2.savefig(figure_file_name, format="pdf", bbox_inches="tight")
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

    # plotVisuals1(str(data_directory_path.joinpath("figure_data.json").resolve()),
    #              str(figure_directory_path.joinpath("figure.pdf").resolve()),
    #              save_fig)

    plotVisuals2(str(data_directory_path.joinpath("figure_data.json").resolve()),
                 str(figure_directory_path.joinpath("figure.pdf").resolve()),
                 save_fig, False)

    plt.show(block=True)


if __name__ == "__main__":
    main()