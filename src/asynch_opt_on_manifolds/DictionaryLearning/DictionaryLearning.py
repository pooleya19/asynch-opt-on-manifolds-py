import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from pathlib import Path

from asynch_opt_on_manifolds.Utils.Manifolds import EuclideanManifold, ObliqueManifold, ProductManifold
from asynch_opt_on_manifolds.Utils.ManifoldUtils import copy_manifold_point
from asynch_opt_on_manifolds.Utils.Agents import Agent, SynchAgent


def calculate_euclidean_grad(agent):
    Y, eps, lam = agent.data

    Y_minus_DC = Y.copy()
    for j in range(len(agent.x)):
        if agent.x[j] is None:
            return None
        Dj, Cj = agent.x[j]
        Y_minus_DC -= Dj@Cj

    Di, Ci = agent.x[agent.i]
    euc_grad_Di = -2 * Y_minus_DC @ Ci.T
    euc_grad_Ci = -2 * Di.T @ Y_minus_DC + lam * Ci / np.sqrt(Ci * Ci + eps)
    euc_grad_xi = [euc_grad_Di, euc_grad_Ci]
    return euc_grad_xi


def calculate_f(data, x):
    Y, eps, lam = data

    Y_minus_DC = Y.copy()
    f = 0
    for j in range(len(x)):
        Dj, Cj = x[j]
        Y_minus_DC -= Dj@Cj
        f += lam * np.sum(np.sqrt(Cj * Cj + eps))

    f += np.linalg.norm(Y_minus_DC, "fro")**2
    return f


def run_sim(parameters=None):
    time_start = time.time()

    # Define parameters
    if parameters is None:
        rng_seed = 0
        N = 10
        n = 50
        B_comm = 100
        B_comp = 100
        kmax = 50000
        gamma = 0.001
        gamma_synch = 0.001
        lam = 1
        eps = 0.01
        data_file_path = str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset").joinpath("synthetic.npy").resolve())
        show_fig = True
        save_output = False
        save_directory_path = None
        save_file_name = None
    else:
        raise RuntimeError("Not implemented yet.")
    #     parameters = {"rng_seed": rng_seed,
    #                   "N": N,
    #                   "B_comm": B_comm,
    #                   "B_comp": B_comp,
    #                   "kmax": kmax,
    #                   "lam": lam,
    #                   "lam_synch": lam_synch,
    #                   "alpha": alpha,
    #                   "graph_edge_probability": graph_edge_probability,
    #                   "world_size": world_size,
    #                   "show_fig": show_fig,
    #                   "save_output": save_output,
    #                   "save_directory_path": save_directory_path,
    #                   "save_file_name": save_file_name,
    #                   }
    # else:
    #     if "B_comm" not in parameters:
    #         parameters["B_comm"] = parameters["B"]
    #     if "B_comp" not in parameters:
    #         parameters["B_comp"] = parameters["B"]
            
    #     rng_seed = parameters["rng_seed"]
    #     N = parameters["N"]
    #     B_comm = parameters["B_comm"]
    #     B_comp = parameters["B_comp"]
    #     kmax = parameters["kmax"]
    #     lam = parameters["lam"]
    #     lam_synch = parameters["lam_synch"]
    #     alpha = parameters["alpha"]
    #     graph_edge_probability = parameters["graph_edge_probability"]
    #     world_size = parameters["world_size"]
    #     show_fig = parameters["show_fig"]
    #     save_output = parameters["save_output"]
    #     save_directory_path = parameters["save_directory_path"]
    #     save_file_name = parameters["save_file_name"]

    render_iteration_mod = 100
    

    # Init RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Load data
    Y = np.load(data_file_path)

    # Generate initial solution value, from Y
    random_samples = rng.choice(Y.shape[1], size=n, replace=False)
    D = Y[:,random_samples].copy()
    D = D / np.linalg.norm(D, axis=0)
    C = np.linalg.pinv(D) @ Y

    # Generate partitions
    ns = n // N + ((n % N) > np.arange(N))
    ns_cumsum = np.concatenate(([0], np.cumsum(ns)))

    # Calculate initial solution values
    x_initial = []
    for i in range(N):
        Di = D[:,ns_cumsum[i]:ns_cumsum[i+1]]
        Ci = C[ns_cumsum[i]:ns_cumsum[i+1],:]
        xi = [Di, Ci]
        x_initial.append(xi)

    # Initialize agents
    agents = []
    agents_synch = []
    data = (Y, eps, lam)
    for i in range(N):
        agents.append(Agent(i, ProductManifold([ObliqueManifold(), EuclideanManifold()]), x_initial, gamma, data, calculate_euclidean_grad))
        agents_synch.append(SynchAgent(i, ProductManifold([ObliqueManifold(), EuclideanManifold()]), x_initial, gamma_synch, data, calculate_euclidean_grad))

    # Initialize mailboxes
    mailbox_outgoing = copy_manifold_point(x_initial)
    mailbox_synch_incoming = [[] for i in range(N)]

    # Initialize iterations at which agents compute, send, and receive
    iteration_to_comp = []
    iteration_to_send = []
    iteration_to_recv = []
    for i in range(N):
        iteration_to_comp.append(rng.integers(low=1,high=B_comp, endpoint=True))
        iteration_to_send.append(rng.integers(low=1,high=(B_comm+1)//2, endpoint=True))
        iteration_to_recv.append(rng.integers(low=1,high=(B_comm+1)//2, endpoint=True))

    # Init plot data
    data_iteration = []
    data_f = []
    data_f_synch = []
    if show_fig:
        ax1_y_min = 1e10
        ax1_y_max = 0

        fig, (ax1) = plt.subplots(1, 1, figsize=(8,5))

        plot_f = ax1.plot([], [], "-r", label="f (asynch)")[0]
        plot_f_synch = ax1.plot([], [], "-b", label="f (synch)")[0]
        ax1.set(xlabel="Iteration", ylabel="Objective Function", yscale="log", title="F (asynch): %.2E\nF (synch): %.2E" % (0,0))
        ax1.legend(loc="lower left")

        plt.show(block=False)

    # Init map for tracking x_synch
    x_synch_map = {}

    # Run simulation
    for k in range(kmax):
        # Send
        for i in range(N):
            if iteration_to_send[i] == k:
                iteration_to_send[i] = k + rng.integers(low=1,high=(B_comm+1)//2, endpoint=True)
                # Let asynch agents send
                mailbox_outgoing[i] = copy_manifold_point(agents[i].x[i])
                # Let synch agents send
                if not agents_synch[i].sent_current:
                    agents_synch[i].sent_current = True
                    for j in range(N):
                        if j != i:
                            message = (i, agents_synch[i].current_iteration, copy_manifold_point(agents_synch[i].x[i]))
                            mailbox_synch_incoming[j].append(message)

        # Recieve
        for i in range(N):
            if iteration_to_recv[i] == k:
                iteration_to_recv[i] = k + rng.integers(low=1,high=(B_comm+1)//2, endpoint=True)
                # Let asynch agents receive
                for j in range(N):
                    if i != j:
                        agents[i].x[j] = copy_manifold_point(mailbox_outgoing[j])
                # Let synch agents receive
                for message in mailbox_synch_incoming[i]:
                    j, message_iteration, message_xj = message
                    if message_iteration < agents_synch[i].current_iteration:
                        pass
                    elif message_iteration == agents_synch[i].current_iteration:
                        agents_synch[i].x[j] = copy_manifold_point(message_xj)
                    elif message_iteration == agents_synch[i].current_iteration + 1:
                        agents_synch[i].next_x[j] = copy_manifold_point(message_xj)
                    else:
                        pass
                        # print("Error: Agent %g at iteration %g received a message from agent %g, timestamped at iteration %g." % (i, agents_synch[i].current_iteration, j, message_iteration))
                        # return
                mailbox_synch_incoming[i] = []

        # Compute
        for i in range(N):
            if iteration_to_comp[i] == k:
                iteration_to_comp[i] = k + rng.integers(low=1,high=B_comp, endpoint=True)
                agents[i].compute()
                agents_synch[i].compute()

        # Generate analytical asynch true x
        x = []
        for i in range(N):
            x.append(copy_manifold_point(agents[i].x[i]))

        # Update analytical synch true x map
        min_x_synch_iteration = 1e10
        for i in range(N):
            synch_iteration_i = agents_synch[i].current_iteration
            synch_xi = copy_manifold_point(agents_synch[i].x[i])
            min_x_synch_iteration = min(min_x_synch_iteration, synch_iteration_i)
            if synch_iteration_i not in x_synch_map:
                x_synch_map[synch_iteration_i] = [None for _ in range(N)]
            x_synch_map[synch_iteration_i][i] = synch_xi

        # Remove outdated synch x from map
        for map_k in list(x_synch_map.keys()):
            if map_k < min_x_synch_iteration:
                del x_synch_map[map_k]

        x_synch = x_synch_map[min_x_synch_iteration]

        iteration_synch = min_x_synch_iteration

        # Calculate f asynch and f synch
        f = calculate_f(data, x)
        f_synch = calculate_f(data, x_synch)

        # Update plots
        data_iteration.append(k)
        data_f.append(f)
        data_f_synch.append(f_synch)
        if show_fig:
            ax1_y_min = min(ax1_y_min, min(f, f_synch))
            ax1_y_max = max(ax1_y_max, max(f, f_synch))

            if k % render_iteration_mod == 0 and ax1_y_min < ax1_y_max:
                print("Iteration %4g: f = %.4E" % (k, f))
                print("\tSynch Iteration %g: f_synch = %.4E" % (iteration_synch, f_synch))
                # print("\tSynch Iterations: %s" % (str([agents_synch[i].current_iteration for i in range(N)])))
                # print("\t|qi|: %s" % (str([np.linalg.norm(x[i][1]).round(2).item() for i in range(N)])))

                plot_f.set(xdata=data_iteration, ydata=data_f)
                plot_f_synch.set(xdata=data_iteration, ydata=data_f_synch)
                ax1.set(xlim=(0,k), ylim=(ax1_y_min,ax1_y_max), title="F (asynch): %.2E\nF (synch): %.2E" % (f,f_synch))

                fig.canvas.draw()
                fig.canvas.flush_events()

    time_end = time.time()
    time_elapsed = time_end - time_start

    if show_fig:
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

        output_data = {"parameters":parameters, "iterations":data_iteration, "f":data_f, "f_synch":data_f_synch}
        with open(file_path, "w") as file:
            json.dump(output_data, file)
    return time_elapsed

if __name__ == "__main__":
    run_sim()