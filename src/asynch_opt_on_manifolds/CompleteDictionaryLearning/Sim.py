import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from pathlib import Path
import cv2

from asynch_opt_on_manifolds.Utils.Manifolds import ObliqueManifold, ProductManifold
from asynch_opt_on_manifolds.Utils.ManifoldUtils import freeze_manifold_point
from asynch_opt_on_manifolds.Utils.Agents import Agent, SynchAgent
from asynch_opt_on_manifolds.Utils.SimUtils import runSimCentralizedRBB
from scipy.optimize import linear_sum_assignment

def calculate_euclidean_grad(agent):
    Y, beta = agent.data
    _, num_samples = Y.shape

    for j in range(len(agent.x)):
        if agent.x[j] is None:
            return None

    A_all = np.hstack(agent.x)
    Ai = agent.x[agent.i]
    euc_grad_xi = -1/num_samples * Y @ np.power(Y.T @ Ai, 3) + beta * (A_all @ (A_all.T @ Ai) - Ai)
    return euc_grad_xi


def calculate_f(data, x):
    Y, beta = data
    dim, num_samples = Y.shape
    A_all = np.hstack(x)
    f = -1/(4*num_samples) * np.sum(np.power(A_all.T @ Y, 4)) + beta/4 * np.linalg.norm(A_all.T @ A_all - np.eye(dim), "fro")**2
    return f


def calculate_norm_grad(data, x):
    Y, beta = data
    _, num_samples = Y.shape
    A_all = np.hstack(x)
    euc_grad = -1/num_samples * Y @ np.power(Y.T @ A_all, 3) + beta * (A_all @ (A_all.T @ A_all) - A_all)
    rie_grad = euc_grad - A_all * np.sum(A_all*euc_grad, axis=0, keepdims=True)
    norm_rie_grad = np.linalg.norm(rie_grad, "fro")
    return norm_rie_grad


def calculate_sparsity(x, threshold_percent, data):
    Y, _ = data
    A_all = np.hstack(x)
    C = A_all.T @ Y
    return 1 - np.mean(np.abs(C) > threshold_percent * np.max(np.abs(C), axis=0, keepdims=True))

    # l1_l2_ratio = np.linalg.norm(C, ord=1, axis=0, keepdims=True)/np.linalg.norm(C, ord=2, axis=0, keepdims=True)
    # return np.mean((C.shape[0] - l1_l2_ratio) / (C.shape[0] - 1))


def calculate_max_coherence(x):
    A = np.hstack(x)
    return np.max(np.abs(A.T@A - np.eye(A.shape[0])))


def run_sim(parameters):
    time_start = time.time()

    # Load parameters
    rng_seed = parameters["rng_seed"]
    N = parameters["N"]
    B_comp1 = parameters["B_comp1"]
    B_comp2 = parameters["B_comp2"]
    B_comm = parameters["B_comm"]
    kmax = parameters["kmax"]
    eta = parameters["eta"]
    eta_synch = parameters["eta_synch"]
    beta = parameters["beta"]
    sparsity_threshold = parameters["sparsity_threshold"]
    m = parameters["m"]
    r = parameters["r"]
    data_file_name = parameters["data_file_name"]
    show_fig = parameters["show_fig"]
    save_final_fig_data = parameters["save_final_fig_data"]
    save_final_fig_data_name = parameters["save_final_fig_data_name"]
    save_video = parameters["save_video"]
    save_video_name = parameters["save_video_name"]
    save_data = parameters["save_data"]
    save_data_name = parameters["save_data_name"]

    # Define parameters
    render_fig = show_fig or save_video
    compute_metrics_mod = 50
    render_mod = 100
    # recon_indices = [1,5,6,7,9,11,15,16,20]
    recon_indices = list(range(10))
    recon_centralized = False

    # Init RNG
    rng_sim = np.random.default_rng(seed=rng_seed)

    # Load data
    Y = np.load(data_file_name)
    dim_raw, num_samples = Y.shape

    # Preprocess data
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    Y_cov = 1/num_samples * (Y - Y_mean) @ (Y - Y_mean).T
    V, U = np.linalg.eigh(Y_cov)
    print("Percent Variance Explained = %.2f" % (np.sum(V[dim_raw-r:dim_raw])/np.sum(V) * 100))
    Vr = V[dim_raw-r:dim_raw]
    Ur = U[:,dim_raw-r:dim_raw]
    Z = np.diag(np.power(Vr,-1/2)) @ Ur.T
    Ypp = Z @ (Y - Y_mean)
    dim = r
    pp_error = np.linalg.norm(1/num_samples * Ypp @ Ypp.T - np.eye(r))
    assert pp_error <= 1e-10, "Data pre-processed incorrectly (error = %.2E)" % pp_error
    Zinv = Ur @ np.diag(np.power(Vr,1/2))

    data = (Ypp, beta)
    atom_indices = list(range(r))
    img_side = int(round(np.sqrt(Y.shape[0])))
    assert img_side**2 == Y.shape[0], "Y must contain square images."
    img_shape = (img_side,img_side)

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

    # Solve centralized problem
    centralized_min_grad_norm = 1e-10
    x_star, f_star = runSimCentralizedRBB(ObliqueManifold(), x_initial, eta_synch, data, calculate_euclidean_grad, calculate_f, centralized_min_grad_norm)
    A_star = np.hstack(x_star)
    C_star = A_star.T @ Ypp[:,recon_indices]
    # print("Finished solving centralized problem! f_star = %.2E" % f_star)

    # Initialize agents
    agents = []
    agents_synch = []
    for i in range(N):
        agents.append(Agent(i, ObliqueManifold(), x_initial, eta, data, calculate_euclidean_grad))
        agents_synch.append(SynchAgent(i, ObliqueManifold(), x_initial, eta_synch, data, calculate_euclidean_grad))

    product_manifold = ProductManifold([ObliqueManifold() for i in range(N)])
    # initial_rie_grad = product_manifold.proj(x_initial, [calculate_euclidean_grad(agents[i]) for i in range(N)])
    # print("Initial grad norm = %.2E" % (product_manifold.norm(initial_rie_grad)))

    # Initialize mailboxes
    mailbox_outgoing = list(x_initial)
    mailbox_synch_incoming = [[] for i in range(N)]

    # Compute B_comps
    B_comp = []
    for i in range(N):
        B_comp_i = B_comp1 if i < N/2 else B_comp2
        B_comp.append(B_comp_i)

    # Initialize iterations at which agents compute, send, and receive
    iteration_to_comp = []
    iteration_to_send = []
    iteration_to_recv = []
    for i in range(N):
        iteration_to_comp.append(rng_sim.integers(low=1,high=B_comp[i], endpoint=True))
        iteration_to_send.append(rng_sim.integers(low=1,high=(B_comm+1)//2, endpoint=True))
        iteration_to_recv.append(rng_sim.integers(low=1,high=(B_comm+1)//2, endpoint=True))

    # Initialize lingering metrics
    x = list(x_initial)
    x_synch = list(x_initial)
    f = calculate_f(data, x)
    f_synch = calculate_f(data, x_synch)
    norm_grad = calculate_norm_grad(data, x)
    norm_grad_synch = calculate_norm_grad(data, x_synch)
    sparsity = 100*calculate_sparsity(x, sparsity_threshold, data)
    sparsity_synch = 100*calculate_sparsity(x_synch, sparsity_threshold, data)
    coherence_star = calculate_max_coherence(x_star)
    coherence = calculate_max_coherence(x)
    coherence_synch = calculate_max_coherence(x_synch)
    iteration_synch = 0
    changed_x = False
    changed_x_synch = False

    # Init plot data
    data_iteration = []
    data_f = []
    data_f_synch = []
    data_norm_grad = []
    data_norm_grad_synch = []
    data_sparsity = []
    data_sparsity_synch = []
    data_coherence_star = []
    data_coherence = []
    data_coherence_synch = []
    ax1_y_min = 1e10
    ax1_y_max = 0
    ax2_y_min = 1e10
    ax2_y_max = 0
    if render_fig:
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18,5))
        
        plot_f = ax1.plot([], [], "-r", label="f (asynch)")[0]
        plot_f_synch = ax1.plot([], [], "-b", label="f (synch)")[0]
        ax1.set(xlabel="Iteration", ylabel="Objective Function", yscale="log", title="F (asynch): %.2E\nF (synch): %.2E" % (0,0))
        ax1.legend(loc="lower left")
        
        plot_norm_grad = ax2.plot([], [], "-r", label="norm grad (asynch)")[0]
        plot_norm_grad_synch = ax2.plot([], [], "-b", label="norm grad (synch)")[0]
        ax2.set(xlabel="Iteration", ylabel="Gradient Norm", yscale="log", title="norm grad (asynch): %.2E\nnorm grad (synch): %.2E" % (0,0))
        ax2.legend(loc="lower left")
        
        plot_sparsity = ax3.plot([], [], "-r", label="sparsity (asynch)")[0]
        plot_sparsity_synch = ax3.plot([], [], "-b", label="sparsity (synch)")[0]
        ax3.set(xlabel="Iteration", ylabel="Sparsity", ylim=(0,100), title="sparsity (asynch): %.2E\nsparsity (synch): %.2E" % (0,0))
        ax3.legend(loc="upper left")

        plot_max_coherence_star = ax4.plot([], [], "-g", label="coherence (star)")[0]
        plot_max_coherence = ax4.plot([], [], "-r", label="coherence (asynch)")[0]
        plot_max_coherence_synch = ax4.plot([], [], "-b", label="coherence (synch)")[0]
        ax4.set(xlabel="Iteration", ylabel="Max Coherence", ylim=(0,1.1), title="Coherence:\nStar=%.2f\nAsynch=%.2f\nSynch=%.2f" % (0,0,0))
        ax4.legend(loc="upper left")

        # Init reconstructions
        v_min_Y = np.min(np.abs(Y))
        v_max_Y = np.max(np.abs(Y))
        fig2 = plt.figure(figsize=(6,5), layout="constrained")
        subfigs2 = fig2.subfigures(1, 3 + recon_centralized, wspace=0.1)
        recon_images = {}
        # recon_rows = int(np.ceil(np.sqrt(len(recon_indices))))
        recon_rows = 5
        recon_cols = int(np.ceil(len(recon_indices)/recon_rows))
        name_asynch = "Reconstruction\n(Algorithm 1)"
        name_synch = "Reconstruction\n(RGD)"
        if recon_centralized:
            recon_names = ["Original", "Centralized", name_asynch, name_synch]
        else:
            recon_names = ["Original", name_asynch, name_synch]
        for subfig, name in zip(subfigs2, recon_names):
            subfig.suptitle(name)
            # subfig.supylabel(name)
            axes = subfig.subplots(recon_rows, recon_cols)
            handles = []
            for k, ax in enumerate(axes.flat):
                ax.set_axis_off()
                if k < len(recon_indices):
                    handles.append(ax.imshow(np.zeros(img_shape), cmap="gray", interpolation="nearest", clim=(v_min_Y,v_max_Y)))
            recon_images[name] = handles

        # Original and top-m centralization reconstruction
        for i in range(len(recon_indices)):
            img_og = np.reshape(Y[:,[recon_indices[i]]], img_shape)
            recon_images["Original"][i].set(data=img_og)

        if recon_centralized:
            Cm_star = np.zeros_like(C_star)
            for i in range(len(recon_indices)):
                index_i_star = np.argpartition(np.abs(C_star[:,i]),-m)[-m:]
                Cm_star[index_i_star,i] = C_star[index_i_star,i]

            for i in range(len(recon_indices)):
                img_star = np.reshape(Zinv @ A_star @ Cm_star[:,[i]] + Y_mean, img_shape)
                recon_images["Centralized"][i].set(data=img_star)

        # Init atoms
        fig3 = plt.figure(figsize=(18,6), layout="constrained")
        subfigs3 = fig3.subfigures(1, 3, wspace=0.03)
        atom_images = []
        atom_rows = int(np.ceil(np.sqrt(len(atom_indices))))
        atom_cols = int(np.ceil(len(atom_indices)/atom_rows))
        for subfig, name in zip(subfigs3, ["Centralized", "Asynch", "Synch"]):
            subfig.suptitle(name)
            axes = subfig.subplots(atom_rows, atom_cols)
            handles = []
            for k, ax in enumerate(axes.flat):
                ax.set_axis_off()
                if k < r:
                    handles.append(ax.imshow(np.zeros(img_shape), cmap="gray", interpolation="nearest"))
            atom_images.append(handles)

        fig1.canvas.draw()
        fig1.canvas.flush_events()
        fig2.canvas.draw()
        fig2.canvas.flush_events()
        fig3.canvas.draw()
        fig3.canvas.flush_events()

        if show_fig:
            plt.show(block=False)
            plt.pause(0.01)

    # Init video
    if save_video:
        video_width, video_height = fig2.canvas.get_width_height(physical=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        os.makedirs(os.path.dirname(save_video_name), exist_ok=True)
        video = cv2.VideoWriter(save_video_name, fourcc, 15.0, (video_width, video_height))


    # Init map for tracking x_synch
    x_synch_map = {0:list(x_initial)}

    # Run simulation
    for k in range(kmax):
        # Send
        for i in range(N):
            if iteration_to_send[i] == k:
                iteration_to_send[i] = k + rng_sim.integers(low=1,high=(B_comm+1)//2, endpoint=True)
                # Let asynch agents send
                mailbox_outgoing[i] = agents[i].x[i]
                # Let synch agents send
                if not agents_synch[i].sent_current:
                    agents_synch[i].sent_current = True
                    message = (i, agents_synch[i].current_iteration, agents_synch[i].x[i])
                    for j in range(N):
                        if j != i:
                            mailbox_synch_incoming[j].append(message)

        # Receive
        for i in range(N):
            if iteration_to_recv[i] == k:
                iteration_to_recv[i] = k + rng_sim.integers(low=1,high=(B_comm+1)//2, endpoint=True)
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
                iteration_to_comp[i] = k + rng_sim.integers(low=1,high=B_comp[i], endpoint=True)
                agent_updated_this_k = True
                agents[i].compute()
                agent_synch_updated_this_k |= agents_synch[i].compute()

        min_x_synch_iteration = agents_synch[0].current_iteration
        for i in range(1,N):
            min_x_synch_iteration = min(min_x_synch_iteration, agents_synch[i].current_iteration)

        if agent_synch_updated_this_k:
            changed_x_synch = True

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
            assert len(x_synch_map) <= N, "Length of x_synch_map exceeds expected length (current length = %g)" % len(x_synch_map)

        if agent_updated_this_k:
            changed_x = True

        # Compute metrics
        if k % compute_metrics_mod == 0:
            # Check if x needs to be recomputed
            if changed_x:
                changed_x = False
                # Recompute x
                for i in range(N):
                    x[i] = agents[i].x[i]

                # Recompute f
                f = calculate_f(data, x)
                norm_grad = calculate_norm_grad(data, x)
                sparsity = 100*calculate_sparsity(x, sparsity_threshold, data)
                coherence = calculate_max_coherence(x)
            
            # Check if x_synch needs to be recomputed
            if changed_x_synch:
                changed_x_synch = False
                # Recompute x_synch
                x_synch = x_synch_map[min_x_synch_iteration]
                iteration_synch = min_x_synch_iteration

                # Recompute f_synch
                f_synch = calculate_f(data, x_synch)
                norm_grad_synch = calculate_norm_grad(data, x_synch)
                sparsity_synch = 100*calculate_sparsity(x_synch, sparsity_threshold, data)
                coherence_synch = calculate_max_coherence(x_synch)

            data_iteration.append(k)
            data_f.append(f)
            data_f_synch.append(f_synch)
            data_norm_grad.append(norm_grad)
            data_norm_grad_synch.append(norm_grad_synch)
            data_sparsity.append(sparsity)
            data_sparsity_synch.append(sparsity_synch)
            data_coherence_star.append(coherence_star)
            data_coherence.append(coherence)
            data_coherence_synch.append(coherence_synch)
            ax1_y_min = min(ax1_y_min, min(f-f_star, f_synch-f_star))
            ax1_y_max = max(ax1_y_max, max(f-f_star, f_synch-f_star))
            ax2_y_min = min(ax2_y_min, min(norm_grad, norm_grad_synch))
            ax2_y_max = max(ax2_y_max, max(norm_grad, norm_grad_synch))

        # Update plots
        if render_fig and k % render_mod == 0 and k>0:
            print("Iteration %g, Synch Iteration %g:" % (k, iteration_synch))
            print("\tf       - f_star = %.2E,\t norm_grad       = %.2E" % (f-f_star, norm_grad))
            print("\tf_synch - f_star = %.2E,\t norm_grad_synch = %.2E" % (f_synch-f_star, norm_grad_synch))
            if ax1_y_min < ax1_y_max:
                plot_f.set(xdata=data_iteration, ydata=np.array(data_f)-f_star)
                plot_f_synch.set(xdata=data_iteration, ydata=np.array(data_f_synch)-f_star)
                ax1.set(xlim=(0,k), ylim=(ax1_y_min,ax1_y_max), title="F (asynch): %.2E\nF (synch): %.2E" % (f,f_synch))

            if ax2_y_min < ax2_y_max:
                plot_norm_grad.set(xdata=data_iteration, ydata=np.array(data_norm_grad))
                plot_norm_grad_synch.set(xdata=data_iteration, ydata=np.array(data_norm_grad_synch))
                ax2.set(xlim=(0,k), ylim=(ax2_y_min,ax2_y_max), title="norm grad (asynch): %.2E\nnorm grad (synch): %.2E" % (norm_grad,norm_grad_synch))

            plot_sparsity.set(xdata=data_iteration, ydata=np.array(data_sparsity))
            plot_sparsity_synch.set(xdata=data_iteration, ydata=np.array(data_sparsity_synch))
            ax3.set(xlim=(0,k), title="sparsity (asynch): %.2E\nsparsity (synch): %.2E" % (sparsity,sparsity_synch))

            plot_max_coherence_star.set(xdata=data_iteration, ydata=np.array(data_coherence_star))
            plot_max_coherence.set(xdata=data_iteration, ydata=np.array(data_coherence))
            plot_max_coherence_synch.set(xdata=data_iteration, ydata=np.array(data_coherence_synch))
            ax4.set(xlim=(0,k), title="Coherence:\nStar=%.2f\nAsynch=%.2f\nSynch=%.2f" % (coherence_star,coherence,coherence_synch))

            # Update reconstructions
            A = np.hstack(x)
            C = A.T @ Ypp[:,recon_indices]
            A_synch = np.hstack(x_synch)
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

            # Update atom reconstructions
            _, index = linear_sum_assignment(-np.abs(A_star.T @ A))
            _, index_synch = linear_sum_assignment(-np.abs(A_star.T @ A_synch))
            for i in atom_indices:
                a_star = np.reshape(Zinv @ A_star[:,i], img_shape)
                a_star *= np.sign(a_star.flat[np.argmax(np.abs(a_star))])
                v_star = np.max(np.abs(a_star))
                atom_images[0][i].set(data=a_star, clim=(-v_star,v_star))

                a = np.reshape(Zinv @ A[:,index[i]], img_shape)
                a *= np.sign(a.flat[np.argmax(np.abs(a))])
                v = np.max(np.abs(a))
                atom_images[1][i].set(data=a, clim=(-v,v))
                
                a_synch = np.reshape(Zinv @ A_synch[:,index_synch[i]], img_shape)
                a_synch *= np.sign(a_synch.flat[np.argmax(np.abs(a_synch))])
                v_synch = np.max(np.abs(a_synch))
                atom_images[2][i].set(data=a_synch, clim=(-v_synch,v_synch))


            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig2.canvas.draw()
            fig2.canvas.flush_events()
            fig3.canvas.draw()
            fig3.canvas.flush_events()

            if save_video:
                img = np.asarray(fig2.canvas.buffer_rgba())[:,:,:3]
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img_bgr)

    # Post run, solve centralized from both solutions
    _, f_star1 = runSimCentralizedRBB(ObliqueManifold(), x, eta_synch, data, calculate_euclidean_grad, calculate_f, centralized_min_grad_norm)
    _, f_star2 = runSimCentralizedRBB(ObliqueManifold(), x_synch, eta_synch, data, calculate_euclidean_grad, calculate_f, centralized_min_grad_norm)
    f_star = min(f_star, min(f_star1, f_star2))
        

    time_end = time.time()
    time_elapsed = time_end - time_start

    if save_final_fig_data:
        os.makedirs(os.path.dirname(save_final_fig_data_name), exist_ok=True)
        final_fig_data = {"Y" : Y.tolist(),
                          "Ypp" : Ypp.tolist(),
                          "Y_mean" : Y_mean.tolist(),
                          "img_shape" : img_shape,
                          "Zinv" : Zinv.tolist(),
                          "A" : np.hstack(x).tolist(),
                          "A_synch" : np.hstack(x_synch).tolist()}
        
        with open(save_final_fig_data_name, "w") as file:
            json.dump(final_fig_data, file)
        

    if save_video:
        video.release()

    if save_data:
        os.makedirs(os.path.dirname(save_data_name), exist_ok=True)

        output_data = {"parameters" : parameters,
                       "iterations" : data_iteration, 
                       "f" : data_f, 
                       "f_synch" : data_f_synch, 
                       "f_star" : f_star,
                       "norm_grad" : data_norm_grad,
                       "norm_grad_synch" : data_norm_grad_synch,
                       "time_elapsed" : time_elapsed,
                       "dtype":str(Y.dtype)}
        
        with open(save_data_name, "w") as file:
            json.dump(output_data, file)

    if show_fig:
        print("Finished simulation. Close figure to quit.")
        plt.show(block=True)

    return time_elapsed


if __name__ == "__main__":
    default_parameters = {
        "rng_seed" : 0,
        "N" : 8,
        "B_comp1" : 2,
        "B_comp2" : 10,
        "B_comm" : 1000,
        "kmax" : 10000,
        "eta" : 0.01,
        "eta_synch" : 0.01,
        "beta" : 20,
        "sparsity_threshold" : 0.01,
        "m" : 5,
        "r" : 83,
        "data_file_name" : str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_dataset").joinpath("fashion_mnist.npy").resolve()),
        "show_fig" : True,
        "save_final_fig_data" : True,
        "save_final_fig_data_name" : str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_learning").joinpath("figure_data.json")),
        "save_video" : True,
        "save_video_name" : str(Path(__file__).parents[3].joinpath("figures").joinpath("dictionary_learning").joinpath("video.mp4")),
        "save_data" : False,
        "save_data_name" : str(Path(__file__).parents[3].joinpath("data").joinpath("dictionary_learning").joinpath("data.json")),
    }

    from threadpoolctl import threadpool_limits
    with threadpool_limits(8, "blas"):
        run_sim(default_parameters)