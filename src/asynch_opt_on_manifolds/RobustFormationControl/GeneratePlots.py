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
from scipy.stats import gmean

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


def plotExperiment(figsize, data_directory_path, figure_file_name, varied_parameter_name, varied_parameter_label, fstar, save_figure):
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
    ax.set(xscale="log")

    # Init speedups
    speedups = []

    # Parse through all runs
    for varied_parameter_index in range(num_varied_parameter_values):
        varied_parameter_value = varied_parameter_values[varied_parameter_index]

        # Get iterations
        iterations = []
        length_asynch = 0
        length_synch = 0

        for rng_seed in rng_seed_values:
            iters = np.array(results_dict[(rng_seed,varied_parameter_value)]["iterations"])
            if len(iters) > len(iterations):
                iterations = iters

            f = np.array(results_dict[(rng_seed, varied_parameter_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, varied_parameter_value)]["f_synch"])

            length_asynch = max(length_asynch, len(f))            
            length_synch = max(length_synch, len(f_synch))

        iterations_asynch = iterations[0:length_asynch]
        iterations_synch = iterations[0:length_synch]
        avg_suboptimality = np.zeros((length_asynch))
        avg_suboptimality_synch = np.zeros((length_synch))
        gavg_suboptimality = np.ones((length_asynch))
        gavg_suboptimality_synch = np.ones((length_synch))

        converged_asynch = True
        converged_synch =  True

        for rng_seed in rng_seed_values:
            # Parse run metrics
            f = np.array(results_dict[(rng_seed, varied_parameter_value)]["f"])
            f_synch = np.array(results_dict[(rng_seed, varied_parameter_value)]["f_synch"])
            f_stop = results_dict[(rng_seed, varied_parameter_value)]["parameters"]["f_stop"]

            if f[-1] > f_stop:
                converged_asynch = False
                f = np.concat((f,f[-1]*np.ones((length_asynch-len(f)))))
            else:
                f[f<f_stop] = f_stop
                f = np.concat((f,f_stop*np.ones((length_asynch-len(f)))))

            if f_synch[-1] > f_stop:
                converged_synch = False
                f_synch = np.concat((f_synch,f_synch[-1]*np.ones((length_synch-len(f_synch)))))
            else:
                f_synch[f_synch<f_stop] = f_stop
                f_synch = np.concat((f_synch,f_stop*np.ones((length_synch-len(f_synch)))))

            suboptimality = f - fstar
            suboptimality_synch = f_synch - fstar

            avg_suboptimality += suboptimality
            avg_suboptimality_synch += suboptimality_synch

            gavg_suboptimality *= suboptimality
            gavg_suboptimality_synch *= suboptimality_synch

            ax.plot(iterations_asynch, suboptimality,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=1, alpha=0.1)
            ax.plot(iterations_synch,  suboptimality_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=1, alpha=0.1)

        avg_suboptimality /= num_rng_seed_values
        avg_suboptimality_synch /= num_rng_seed_values

        gavg_suboptimality = np.power(gavg_suboptimality, 1/num_rng_seed_values)
        gavg_suboptimality_synch = np.power(gavg_suboptimality_synch, 1/num_rng_seed_values)

        # ax.plot(iterations, avg_suboptimality,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=2, alpha=1)
        # ax.plot(iterations, avg_suboptimality_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=2, alpha=1)
        ax.plot(iterations_asynch, gavg_suboptimality,       color=colors[varied_parameter_index,:], linestyle="-", linewidth=2, alpha=1)
        ax.plot(iterations_synch,  gavg_suboptimality_synch, color=colors[varied_parameter_index,:], linestyle=":", linewidth=2, alpha=1)

        final_iter_asynch = int(np.max(iterations_asynch))
        final_iter_synch = int(np.max(iterations_synch))

        print("B=%g:\n\tAsynch: %6g\n\t Synch: %6g" % (varied_parameter_value, final_iter_asynch, final_iter_synch))

        if converged_asynch:
            if converged_synch:
                speedups.append(final_iter_synch / final_iter_asynch)
            else:
                speedups.append(np.inf)
        else:
            if converged_synch:
                speedups.append(0)
            else:
                speedups.append(np.nan)

    # Generate legends
    color_handles = [Line2D([], [], color=colors[i,:], linestyle="-", linewidth=2) for i in range(num_varied_parameter_values)]
    color_labels = ["%s=%g"%(varied_parameter_label,varied_parameter_values[i]) for i in range(num_varied_parameter_values)]
    style_handles = [Line2D([], [], color="black", linestyle="-", linewidth=2),
                      Line2D([], [], color="black", linestyle=":", linewidth=2)]
    style_labels = ["Algorithm 1", "RGD"]

    leg1 = ax.legend(style_handles, style_labels, loc="lower left")
    ax.add_artist(leg1)
    fig.canvas.draw()
    bb = leg1.get_window_extent(fig.canvas.get_renderer())
    bb = bb.transformed(ax.transAxes.inverted())
    ax.legend(color_handles, color_labels, loc="lower left", bbox_to_anchor=(0, bb.y1), bbox_transform=ax.transAxes)

    # leg1 = ax.legend(color_handles, color_labels, loc="lower left")
    # ax.add_artist(leg1)
    # fig.canvas.draw()
    # bb = leg1.get_window_extent(fig.canvas.get_renderer())
    # bb = bb.transformed(ax.transAxes.inverted())
    # ax.legend(style_handles, style_labels, loc="lower left", bbox_to_anchor=(bb.x1, 0), bbox_transform=ax.transAxes)

    # speedups = np.floor(speedups)

    print("Vary %s: Speed Ups: %.1f-%.1f" % (varied_parameter_label,np.min(speedups), np.max(speedups)))
    for varied_parameter_index in range(num_varied_parameter_values):
        varied_parameter_value = varied_parameter_values[varied_parameter_index]
        print("\t%s=%g: %.2f" % (varied_parameter_label, varied_parameter_value, speedups[varied_parameter_index]))

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
    # figsize = (10,6.6)
    figsize = (12,3.3)
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
    edge_linewidth = 0.3

    lim_xy = 0.72
    lim_z = 0.5
    # gap = 0.03
    gap = 0
    cam_elev = 18
    cam_azim = -60
    caption_height = 0.07
    # panel_height = (1-gap)/2
    panel_height = 1-gap

    divider_color = 0.8*np.ones((3))
    divider_linewidth = 0.8

    # Define panels
    # panels = [(x_formation, "Desired Formation", (0.0, panel_height + gap)),
    #           (x_initial,   "Initial Poses",     (0.5, panel_height + gap)),
    #           (x,           "Final Poses (Algorithm 1)",       (0.0, 0.0)),
    #           (x_synch,     "Final Poses (RGD)",               (0.5, 0.0))]
    panels = [(x_formation, "Desired Formation", (0.0, 0.0)),
              (x_initial,   "Initial Poses",     (0.25, 0.0)),
              (x,           "Final Poses (Algorithm 1)",       (0.5, 0.0)),
              (x_synch,     "Final Poses (RGD)",               (0.75, 0.0))]

    for x_plot, caption, (x0, y0) in panels:
        # ax = fig2.add_axes([x0, y0+caption_height, 0.5, panel_height-caption_height], projection="3d")
        ax = fig2.add_axes([x0, y0+caption_height, 0.25, panel_height-caption_height], projection="3d")

        # Plot formations
        positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_plot, formation_edges, axes_length)
        ax.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3, clip_on=False)
        ax.add_collection3d(Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth, clip_on=False))
        ax.add_collection3d(Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth, clip_on=False))
        ax.add_collection3d(Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth, clip_on=False))
        ax.add_collection3d(Line3DCollection(edges, colors="gray", linewidths=edge_linewidth, clip_on=False))
        ax.set(xlim=(-lim_xy,lim_xy), ylim=(-lim_xy,lim_xy), zlim=(-lim_z,lim_z))
        # ax.set_box_aspect((1, 1, lim_z/lim_xy), zoom=1.8)
        ax.set_box_aspect((1, 1, lim_z/lim_xy), zoom=1.4)
        ax.view_init(elev=cam_elev, azim=cam_azim)
        ax.axis("off")

        # Plot title
        # fig2.text(x0+0.25, y0+caption_height/2, caption, ha="center", va="center", fontsize=text_fontsize/ieee_width*figsize[0])
        fig2.text(x0+0.125, y0+caption_height/2, caption, ha="center", va="center", fontsize=18)

        # Plot x-y-z axes indicator
        ax_xyz = fig2.add_axes([x0+0.007, y0+caption_height+0.02, 0.07, 0.11], projection="3d")
        for c, (name, color) in enumerate(zip("xyz", "rgb")):
            e = np.zeros(3)
            e[c] = 1
            ax_xyz.add_collection3d(Line3DCollection([[np.zeros(3), e]], colors=color, linewidths=axes_linewidth))
            # ax_xyz.text(1.35*e[0], 1.35*e[1], 1.35*e[2], "$%s$" % name, fontsize=axes_fontsize/ieee_width*figsize[0], ha="center", va="center")
            ax_xyz.text(1.8*e[0], 1.8*e[1], 1.8*e[2], "$%s$" % name, fontsize=15, ha="center", va="center")
        ax_xyz.set(xlim=(-0.2,1.2), ylim=(-0.2,1.2), zlim=(-0.2,1.2))
        ax_xyz.set_box_aspect((1,1,1))
        ax_xyz.view_init(elev=cam_elev, azim=cam_azim)
        ax_xyz.axis("off")
            
    # Plot dividers between panels
    # if show_dividers:
    #     fig2.add_artist(Line2D([0.02, 0.98], [0.5, 0.5], color=divider_color, linewidth=divider_linewidth))
    #     fig2.add_artist(Line2D([0.5, 0.5], [0.02, 0.98], color=divider_color, linewidth=divider_linewidth))

    # Save figure
    if save_figure:
        print("Saving %s... " % figure_file_name, end='', flush=True)
        fig2.savefig(figure_file_name, format="pdf", bbox_inches="tight")
        print("Done!")


def plotStepSizeSweep(data_directory_path):
    # Compress all results and identify unique parameter values
    varied_parameters = ["rng_seed", "eta", "B_comm", "N"]
    unique_parameter_values, results_dict = compress_results(varied_parameters, data_directory_path)
    values_rng_seed, values_eta, values_Bcomm, values_N = unique_parameter_values
    num_per_eta = len(results_dict.keys())//len(values_eta)

    # Init figure
    colors = mpl.colormaps["rainbow"](np.linspace(0, 1, len(values_rng_seed)))[:,0:3] # jet, viridis, rainbow
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5))
    ax1.set(xlabel=r"Step size, $\eta$", ylabel=r"Final Suboptimality,  $f(x(k_{max})) - f^\star$", xscale="log", yscale="log")
    ax2.set(xlabel=r"Step size, $\eta$", ylabel=r"Final Suboptimality,  $f(x(k_{max})) - f^\star$", xscale="log", yscale="log")

    fstar = 0

    # subopts = np.zeros((num_per_eta, len(values_eta)))
    # subopts_i = [0] * len(values_eta)
    # subopts_synch = np.zeros((num_per_eta, len(values_eta)))
    # subopts_i_synch = [0] * len(values_eta)

    convergence_iters = np.inf * np.ones((num_per_eta, len(values_eta)))
    convergence_iters_i = [0] * len(values_eta)
    convergence_iters_synch = np.inf * np.ones((num_per_eta, len(values_eta)))
    convergence_iters_synch_i = [0] * len(values_eta)

    for key in results_dict.keys():
        eta = key[1]
        eta_index = values_eta.index(eta)

        f = results_dict[key]["f"]
        f_synch = results_dict[key]["f_synch"]
        f_stop = results_dict[key]["parameters"]["f_stop"]

        if f[-1] <= f_stop:
            convergence_iters[convergence_iters_i[eta_index], eta_index] = len(f)
        else:
            convergence_iters[convergence_iters_i[eta_index], eta_index] = np.inf
        convergence_iters_i[eta_index] = convergence_iters_i[eta_index] + 1

        if f_synch[-1] <= f_stop:
            convergence_iters_synch[convergence_iters_synch_i[eta_index], eta_index] = len(f_synch)
        else:
            convergence_iters_synch[convergence_iters_synch_i[eta_index], eta_index] = np.inf
        convergence_iters_synch_i[eta_index] = convergence_iters_synch_i[eta_index] + 1

        # ax1.plot(eta, final_f - fstar, "r.")
        # ax2.plot(eta, final_f_synch - fstar, "r.")

    convergence_iters_mean = np.mean(convergence_iters, axis=0)
    convergence_iters_synch_mean = np.mean(convergence_iters_synch, axis=0)

    argmin_con = np.argmin(convergence_iters_mean)
    argmin_con_synch = np.argmin(convergence_iters_synch_mean)
    
    ax1.plot(np.tile(values_eta, (num_per_eta,1)).flatten(), convergence_iters.flatten(), "r.")
    ax1.plot(values_eta, convergence_iters_mean, "b-")
    ax1.plot(values_eta[argmin_con], convergence_iters_mean[argmin_con], "bo")
    ax2.plot(np.tile(values_eta, (num_per_eta,1)).flatten(), convergence_iters_synch.flatten(), "r.")
    ax2.plot(values_eta, convergence_iters_synch_mean, "b-")
    ax2.plot(values_eta[argmin_con_synch], convergence_iters_synch_mean[argmin_con_synch], "bo")

    print("Asynch:")
    for i in range(len(values_eta)):
        print("\t%.1E: %.2E%s" % (values_eta[i], convergence_iters_mean[i], (" (best)" if i==argmin_con else "")))
    print("Synch:")
    for i in range(len(values_eta)):
        print("\t%.1E: %.2E%s" % (values_eta[i], convergence_iters_synch_mean[i], (" (best)" if i==argmin_con_synch else "")))


def calc_fstar(directories):
    fstar = 10**10
    for directory in directories:
        for object in Path(directory).iterdir():
            if object.is_file():
                with open(object, "r") as file:
                    data = json.load(file)
                    data_fstar = data["f_star"]
                    fstar = min(fstar, data_fstar)
    return fstar


def main():
    figsize = (8,4)
    save_fig = True
    # data_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("formation_control")
    # figure_directory_path = Path(__file__).parents[3].joinpath("figures").joinpath("formation_control")
    data_directory_path = Path(__file__).parents[3].joinpath("data").joinpath("formation_control_alpha0")
    figure_directory_path = Path(__file__).parents[3].joinpath("figures").joinpath("formation_control_alpha0")
    figure_directory_path.mkdir(parents=True, exist_ok=True)

    fstar = calc_fstar([str(data_directory_path.joinpath("exp1").resolve())])

    plotExperiment(figsize, 
                   str(data_directory_path.joinpath("exp1").resolve()), 
                   str(figure_directory_path.joinpath("f_vs_B.pdf").resolve()), 
                   "B_comm", "B", fstar, save_fig)

    # plotVisuals1(str(data_directory_path.joinpath("figure_data.json").resolve()),
    #              str(figure_directory_path.joinpath("figure.pdf").resolve()),
    #              save_fig)

    plotVisuals2(str(data_directory_path.joinpath("figure_data.json").resolve()),
                 str(figure_directory_path.joinpath("figure.pdf").resolve()),
                 save_fig, False)

    # plotStepSizeSweep(str(data_directory_path.joinpath("exp_step").resolve()))
    # plotStepSizeSweep(str(data_directory_path.joinpath("exp_step2").resolve()))

    # plt.show(block=True)


if __name__ == "__main__":
    main()