import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from pathlib import Path

from asynch_opt_on_manifolds.Utils.Manifolds import ObliqueManifold
from asynch_opt_on_manifolds.Utils.ManifoldUtils import freeze_manifold_point
from asynch_opt_on_manifolds.Utils.Agents import Agent, SynchAgent


def calculate_euclidean_grad(agent):
    Y, beta = agent.data
    dim, num_samples = Y.shape

    for j in range(len(agent.x)):
        if agent.x[j] is None:
            return None

    A_all = np.hstack([agent.x[j] for j in range(len(agent.x))])
    Ai = agent.x[agent.i]
    euc_grad_xi = -1/num_samples * Y @ np.power(Y.T @ Ai, 3) + beta * (A_all @ (A_all.T @ Ai) - Ai)
    return euc_grad_xi


def calculate_f(data, x):
    Y, beta = data
    dim, num_samples = Y.shape

    A_all = np.hstack(x)

    f = -1/(4*num_samples) * np.sum(np.power(A_all.T @ Y, 4)) + beta/4 * np.linalg.norm(A_all.T @ A_all - np.eye(dim), "fro")**2
    return f


def calculate_sparsity(x, threshold_percent, data):
    Y, _ = data
    A_all = np.hstack(x)
    C = A_all.T @ Y
    return 1 - np.mean(np.abs(C) > threshold_percent * np.max(np.abs(C), axis=0, keepdims=True))

    # l1_l2_ratio = np.linalg.norm(C, ord=1, axis=0, keepdims=True)/np.linalg.norm(C, ord=2, axis=0, keepdims=True)
    # return np.mean((C.shape[0] - l1_l2_ratio) / (C.shape[0] - 1))



def run_sim(parameters):
    time_start = time.time()

    # Define parameters
    rng_seed = parameters["rng_seed"]
    N = parameters["N"]
    B_comm = parameters["B_comm"]
    B_calc = parameters["B_calc"]
    kmax = parameters["kmax"]
    eta = parameters["eta"]
    eta_synch = parameters["eta_synch"]
    beta = parameters["beta"]
    sparsity_threshold = parameters["sparsity_threshold"]
    data_directory_path = parameters["data_directory_path"]
    data_file_name = parameters["data_file_name"]
    show_fig = parameters["show_fig"]
    save_fig = parameters["save_fig"]
    save_output = parameters["save_output"]
    save_directory_path = parameters["save_directory_path"]
    save_file_name = parameters["save_file_name"]
    save_fig_name = parameters["save_fig_name"]

    min_calc_f_iteration_delay = min(100, B_calc)
    render_iteration_mod = min(100, B_calc)
    max_plot_points = 1000
    reconstruction_indices = [4,1,3]
    num_reconstructions = len(reconstruction_indices)
    

    # Init RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Load data
    data_file_path = str(Path(data_directory_path).joinpath(data_file_name).resolve())
    Y = np.load(data_file_path)
    dim, num_samples = Y.shape

    # Generate initial solution value, from Y
    A = np.eye(dim)

    # Generate partitions
    ns = dim // N + ((dim % N) > np.arange(N))
    ns_cumsum = np.concatenate(([0], np.cumsum(ns)))

    # Calculate initial solution values
    x_initial = []
    for i in range(N):
        xi = A[:,ns_cumsum[i]:ns_cumsum[i+1]].copy()
        x_initial.append(xi)
    freeze_manifold_point(x_initial)

    # Initialize agents
    agents = []
    agents_synch = []
    data = (Y, beta)
    for i in range(N):
        agents.append(Agent(i, ObliqueManifold(), x_initial, eta, data, calculate_euclidean_grad))
        agents_synch.append(SynchAgent(i, ObliqueManifold(), x_initial, eta_synch, data, calculate_euclidean_grad))

    # Initialize mailboxes
    mailbox_outgoing = list(x_initial)
    mailbox_synch_incoming = [[] for i in range(N)]

    # Initialize iterations at which agents compute, send, and receive
    iteration_to_comp = []
    iteration_to_send = []
    iteration_to_recv = []
    for i in range(N):
        iteration_to_comp.append(rng.integers(low=1,high=B_calc, endpoint=True))
        iteration_to_send.append(rng.integers(low=1,high=(B_comm+1)//2, endpoint=True))
        iteration_to_recv.append(rng.integers(low=1,high=(B_comm+1)//2, endpoint=True))

    # Init plot data
    data_iteration = []
    data_f = []
    data_f_synch = []
    data_sparsity = []
    data_sparsity_synch = []
    ax1_y_min = 1e10
    ax1_y_max = 0
    if show_fig:
        fig1, (ax1) = plt.subplots(1, 1, figsize=(8,5))

        plot_f = ax1.plot([], [], "-r", label="f (asynch)")[0]
        plot_f_synch = ax1.plot([], [], "-b", label="f (synch)")[0]
        ax1.set(xlabel="Iteration", ylabel="Objective Function", yscale="log", title="F (asynch): %.2E\nF (synch): %.2E" % (0,0))
        ax1.legend(loc="lower left")

        # Init reconstructions
        reconstructions = []
        axes_images = []
        vmin = np.min(Y)
        vmax = np.max(Y)
        fig2, axes2 = plt.subplots(num_reconstructions, 3, figsize=(9/num_reconstructions*3,9))
        axes2[0,0].set(title="True")
        axes2[0,1].set(title="Asynch\nReconstructions")
        axes2[0,2].set(title="Synch\nReconstructions")
        img_side = int(round(np.sqrt(Y.shape[0])))
        assert img_side**2 == Y.shape[0], "Y must contain square images."
        img_shape = (img_side,img_side)
        for i in range(num_reconstructions):
            img = Y[:,reconstruction_indices[i]].reshape(img_shape)
            axes2[i,0].imshow(img, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            axes_image_asynch = axes2[i,1].imshow([[0]], cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            axes_image_synch  = axes2[i,2].imshow([[0]], cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
            axes_images.append([axes_image_asynch, axes_image_synch])
            reconstructions.append([[[0]], [[0]]])
            for j in range(3):
                axes2[i,j].set_axis_off()

        plt.show(block=False)
        plt.pause(0.01)

    # Init map for tracking x_synch
    x_synch_map = {0:list(x_initial)}
    k_of_last_f_calc = -1e10

    # Run simulation
    for k in range(kmax):
        # Send
        for i in range(N):
            if iteration_to_send[i] == k:
                iteration_to_send[i] = k + rng.integers(low=1,high=(B_comm+1)//2, endpoint=True)
                # Let asynch agents send
                mailbox_outgoing[i] = agents[i].x[i]
                # Let synch agents send
                if not agents_synch[i].sent_current:
                    agents_synch[i].sent_current = True
                    message = (i, agents_synch[i].current_iteration, agents_synch[i].x[i])
                    for j in range(N):
                        if j != i:
                            mailbox_synch_incoming[j].append(message)

        # Recieve
        for i in range(N):
            if iteration_to_recv[i] == k:
                iteration_to_recv[i] = k + rng.integers(low=1,high=(B_comm+1)//2, endpoint=True)
                # Let asynch agents receive
                for j in range(N):
                    if i != j:
                        agents[i].x[j] = mailbox_outgoing[j]
                # Let synch agents receive
                for message in mailbox_synch_incoming[i]:
                    j, message_iteration, message_xj = message
                    if message_iteration < agents_synch[i].current_iteration:
                        pass
                    elif message_iteration == agents_synch[i].current_iteration:
                        agents_synch[i].x[j] = message_xj
                    elif message_iteration == agents_synch[i].current_iteration + 1:
                        agents_synch[i].next_x[j] = message_xj
                    else:
                        pass
                        # print("Error: Agent %g at iteration %g received a message from agent %g, timestamped at iteration %g." % (i, agents_synch[i].current_iteration, j, message_iteration))
                        # return
                mailbox_synch_incoming[i] = []

        # Compute
        agent_updated_this_k = False
        agent_synch_updated_this_k = False
        for i in range(N):
            if iteration_to_comp[i] == k:
                iteration_to_comp[i] = k + rng.integers(low=1,high=B_calc, endpoint=True)
                agent_updated_this_k = True
                agents[i].compute()
                agent_synch_updated_this_k |= agents_synch[i].compute()

        min_x_synch_iteration = 1e10
        for i in range(N):
            min_x_synch_iteration = min(min_x_synch_iteration, agents_synch[i].current_iteration)

        if agent_synch_updated_this_k:
            # Update analytical synch true x map
            for i in range(N):
                synch_iteration_i = agents_synch[i].current_iteration
                synch_xi = agents_synch[i].x[i]
                if synch_iteration_i not in x_synch_map:
                    x_synch_map[synch_iteration_i] = [None for _ in range(N)]
                x_synch_map[synch_iteration_i][i] = synch_xi

            # Remove outdated synch x from map
            for map_k in list(x_synch_map.keys()):
                if map_k < min_x_synch_iteration:
                    del x_synch_map[map_k]
            assert len(x_synch_map) < 5, "Length of x_synch_map exceeds expected length (current length = %g)" % len(x_synch_map)

        if agent_updated_this_k and k - k_of_last_f_calc >= min_calc_f_iteration_delay:
            k_of_last_f_calc = k
        
            # Generate analytical asynch true x
            x = []
            for i in range(N):
                x.append(agents[i].x[i])

            # Get analytical synch true x
            x_synch = x_synch_map[min_x_synch_iteration]
            iteration_synch = min_x_synch_iteration

            # Calculate f asynch and f synch
            f = calculate_f(data, x)
            f_synch = calculate_f(data, x_synch)

            # Calculate sparsities
            sparsity = 100*calculate_sparsity(x, sparsity_threshold, data)
            sparsity_synch = 100*calculate_sparsity(x_synch, sparsity_threshold, data)

            # Append new data
            data_iteration.append(k)
            data_f.append(float(f))
            data_f_synch.append(float(f_synch))
            data_sparsity.append(float(sparsity))
            data_sparsity_synch.append(float(sparsity_synch))

            # Update min/max values of f
            ax1_y_min = min(ax1_y_min, min(f, f_synch))
            ax1_y_max = max(ax1_y_max, max(f, f_synch))

            # Update reconstructions
            if show_fig:
                A = np.hstack(x)
                C = A.T @ Y[:,reconstruction_indices]
                A_synch = np.hstack(x_synch)
                C_synch = A_synch.T @ Y[:,reconstruction_indices]

                for i in range(num_reconstructions):
                    img = np.reshape(A @ C[:,i], img_shape)
                    img_synch = np.reshape(A_synch @ C_synch[:,i], img_shape)
                    reconstructions[i][0] = img
                    reconstructions[i][1] = img_synch

        # Update plots
        if show_fig and k % render_iteration_mod == 0 and ax1_y_min < ax1_y_max:
            print("Iteration %g, Synch Iteration %g:" % (k, iteration_synch))
            print("\tf:      = %.2E,  sparsity       = %.2f%%" % (f,       sparsity))
            print("\tf_synch = %.2E,  sparsity_synch = %.2f%%" % (f_synch, sparsity_synch))

            plot_step = max(1,len(data_iteration)//max_plot_points)
            plot_f.set(xdata=data_iteration[::plot_step], ydata=data_f[::plot_step])
            plot_f_synch.set(xdata=data_iteration[::plot_step], ydata=data_f_synch[::plot_step])
            # plot_f.set(xdata=data_iteration, ydata=data_f)
            # plot_f_synch.set(xdata=data_iteration, ydata=data_f_synch)
            ax1.set(xlim=(0,k), ylim=(ax1_y_min,ax1_y_max), title="F (asynch): %.2E\nF (synch): %.2E" % (f,f_synch))

            # Render reconstructions
            for i in range(num_reconstructions):
                axes_images[i][0].set(data=reconstructions[i][0])
                axes_images[i][1].set(data=reconstructions[i][1])

            # Draw figures
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig2.canvas.draw()
            fig2.canvas.flush_events()


    time_end = time.time()
    time_elapsed = time_end - time_start

    if show_fig:
        if save_fig:
            print("Saving figure... ", end='', flush=True)
            Path(save_directory_path).mkdir(parents=True, exist_ok=True)
            fig_path = str(Path(save_directory_path).joinpath(save_fig_name).resolve())
            fig2.savefig(fig_path, format="svg")
            print("Done!")

        print("Finished simulation. Close figure to quit.")
        plt.show(block=True)

    if save_output:
        file_path = os.path.join(save_directory_path, save_file_name)
        keys_descending_length = list(parameters.keys())
        keys_descending_length.sort(key=len, reverse=True)
        for i in range(len(keys_descending_length)):
            key = keys_descending_length[i]
            file_path = file_path.replace("<"+key+">", str(parameters[key]))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        output_data = {
            "parameters":parameters, 
            "iterations":data_iteration, 
            "f":data_f, 
            "f_synch":data_f_synch, 
            "sparsity":data_sparsity, 
            "sparsity_synch":data_sparsity_synch, 
            "time_elapsed":time_elapsed,
            "dtype":str(Y.dtype)
        }
        with open(file_path, "w") as file:
            json.dump(output_data, file)
    return time_elapsed


if __name__ == "__main__":
    parameters = {
        "rng_seed" : 0,
        "N" : 10,
        "B_comm" : 100,
        "B_calc" : 10,
        "kmax" : 5000,
        "eta" : 1,
        'eta_synch' : 1,
        'beta' : 1,
        "lam" : 0.02,
        "eps" : 1e-7,
        "sparsity_threshold" : 0.005,
        "data_directory_path" : Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset"),
        "data_file_name" : "synthetic.npy",
        "show_fig" : True,
        "save_fig" : False,
        "save_output" : False,
        "save_directory_path" : None,
        "save_file_name" : None,
        "save_fig_name" : None,
    }
    from threadpoolctl import threadpool_limits
    with threadpool_limits(8, "blas"):
        run_sim(parameters)