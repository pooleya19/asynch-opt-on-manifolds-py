import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os

from asynch_opt_on_manifolds.Utils.Manifolds import EuclideanManifold, UnitQuaternionManifold, ProductManifold
from asynch_opt_on_manifolds.Utils.ManifoldUtils import freeze_manifold_point
from asynch_opt_on_manifolds.Utils.Agents import Agent, SynchAgent
from asynch_opt_on_manifolds.Utils.RotationUtils import quat_to_rot_mat, partial_rot_mat_partial_quat
from asynch_opt_on_manifolds.RobustFormationControl.Formations import generateFormationHelix
from asynch_opt_on_manifolds.Utils.SimUtils import runSimCentralized


def calculate_euclidean_grad(agent):
    alpha, delta, prior_poses, formation_edges, formation_offsets = agent.data
    N = len(prior_poses)

    ti = agent.x[agent.i][0]
    Ri = quat_to_rot_mat(agent.x[agent.i][1])

    # Calculate gradient f wrt ti
    nabla_ti = 2 * alpha * (ti - prior_poses[agent.i][0])
    for k in range(len(formation_edges)):
        if agent.i == formation_edges[k][0]:
            _,j = formation_edges[k]
            if agent.x[j] is None:
                return None
            tj = agent.x[j][0]
            ttilde_ij = formation_offsets[k]
            rij = Ri.T @ (tj - ti) - ttilde_ij
            nabla_ti -= (1-alpha)/N * delta/np.sqrt(delta**2 + rij.T @ rij) * Ri @ rij
        elif agent.i == formation_edges[k][1]:
            j,_ = formation_edges[k]
            if agent.x[j] is None:
                return None
            ttilde_ji = formation_offsets[k]
            tj = agent.x[j][0]
            Rj = quat_to_rot_mat(agent.x[j][1])
            rji = Rj.T @ (ti - tj) - ttilde_ji
            nabla_ti += (1-alpha)/N * delta/np.sqrt(delta**2 + rji.T @ rji) * Rj @ rji

    # Calculate gradient f wrt Ri
    nabla_Ri = 2 * alpha * (Ri - quat_to_rot_mat(prior_poses[agent.i][1]))
    for k in range(len(formation_edges)):
        if agent.i == formation_edges[k][0]:
            _,j = formation_edges[k]
            if agent.x[j] is None:
                return None
            tj = agent.x[j][0]
            ttilde_ij = formation_offsets[k]
            rij = Ri.T @ (tj - ti) - ttilde_ij
            nabla_Ri += (1-alpha)/N * delta/np.sqrt(delta**2 + rij.T @ rij) * (tj - ti) @ rij.T

    # Calculate partial of Rqi wrt qic, c = [w,x,y,z]
    partials_Ri = partial_rot_mat_partial_quat(agent.x[agent.i][1])

    # Calculate gradient of f wrt qi
    nabla_qi = np.zeros((4,1))
    for c in range(4):
        nabla_qi[c,0] = np.trace(nabla_Ri.T @ partials_Ri[c])

    nabla_xi = [nabla_ti, nabla_qi]
    return nabla_xi


def pseudo_huber(x, delta):
    return delta**2 * (np.sqrt(1 + np.linalg.norm(x)**2/delta**2) - 1)


def calculate_f(data, x):
    alpha, delta, prior_poses, formation_edges, formation_offsets = data
    N = len(prior_poses)

    f = 0
    for i in range(N):
        ti = x[i][0]
        Ri = quat_to_rot_mat(x[i][1])
        f += alpha * (np.linalg.norm(ti - prior_poses[i][0])**2 + np.linalg.norm(Ri - quat_to_rot_mat(prior_poses[i][1]), "fro")**2)

    for k in range(len(formation_edges)):
        i,j = formation_edges[k]
        ti = x[i][0]
        tj = x[j][0]
        Ri = quat_to_rot_mat(x[i][1])
        ttilde_ij = formation_offsets[k]
        rij = Ri.T @ (tj - ti) - ttilde_ij
        f += (1-alpha)/N * pseudo_huber(rij, delta)

    return f


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
    alpha = parameters["alpha"]
    delta = parameters["delta"]
    show_fig = parameters["show_fig"]
    # save_fig = parameters["save_fig"]
    # save_output = parameters["save_output"]
    # save_directory_path = parameters["save_directory_path"]
    # save_file_name = parameters["save_file_name"]
    # save_fig_name = parameters["save_fig_name"]
    
    # Define parameters
    compute_metrics_mod = 10
    render_mod = 100

    # Init RNG
    rng_static = np.random.default_rng(seed=0)
    rng_sim = np.random.default_rng(seed=rng_seed)

    # Generate prior poses
    prior_poses = []
    for i in range(N):
        random_position = rng_static.uniform(low=np.zeros((3,1)), high=np.ones((3,1)), size=(3,1))
        random_quaternion = rng_static.multivariate_normal(mean=np.zeros(4), cov=np.eye(4)).reshape((4,1))
        random_unit_quaternion = random_quaternion / np.linalg.norm(random_quaternion)
        prior_poses.append([random_position, random_unit_quaternion])

    # Generate formation
    formation_positions, formation_edges, formation_offsets = generateFormationHelix(N)
    data = (alpha, delta, prior_poses, formation_edges, formation_offsets)

    # Initialize solution
    x_initial = list(prior_poses)
    freeze_manifold_point(x_initial)

    # Solve centralized problem
    centralized_kmax = 10000
    centralized_min_grad_norm = 1e-15
    print("Solving centralized problem...")
    f_star = runSimCentralized(ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, eta_synch, data, calculate_euclidean_grad, calculate_f, centralized_kmax, centralized_min_grad_norm)
    # f_star = -100
    print("Finished solving centralized problem! f_star = %.2E" % f_star)

    # Initialize agents
    agents = []
    agents_synch = []
    for i in range(N):
        agents.append(Agent(i, ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, eta, data, calculate_euclidean_grad))
        agents_synch.append(SynchAgent(i, ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, eta_synch, data, calculate_euclidean_grad))

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
    iteration_synch = 0
    changed_x = False
    changed_x_synch = False

    # Init plot data
    data_iteration = []
    data_f = []
    data_f_synch = []
    ax1_y_min = 1e10
    ax1_y_max = 0
    if show_fig:
        fig1, (ax1) = plt.subplots(1, 1, figsize=(8,5))

        plot_f = ax1.plot([], [], "-r", label="f (asynch)")[0]
        plot_f_synch = ax1.plot([], [], "-b", label="f (synch)")[0]
        ax1.set(xlabel="Iteration", ylabel="f - fstar", yscale="log", title="F (asynch): %.2E\nF (synch): %.2E" % (0,0))
        ax1.legend(loc="lower left")

        # fig2, (ax2, ax3, ax4, ax5) = plt.subplots(1, 4, figsize=(20,5))

        plt.show(block=False)
        plt.pause(0.01)

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
            
            # Check if x_synch needs to be recomputed
            if changed_x_synch:
                changed_x_synch = False
                # Recompute x_synch
                x_synch = x_synch_map[min_x_synch_iteration]
                iteration_synch = min_x_synch_iteration

                # Recompute f_synch
                f_synch = calculate_f(data, x_synch)

            data_iteration.append(k)
            data_f.append(f)
            data_f_synch.append(f_synch)
            ax1_y_min = min(ax1_y_min, min(f-f_star, f_synch-f_star))
            ax1_y_max = max(ax1_y_max, max(f-f_star, f_synch-f_star))

        # Update plots
        if show_fig and k % render_mod == 0:
            if ax1_y_min < ax1_y_max:
                print("Iteration %g, Synch Iteration %g:" % (k, iteration_synch))
                print("\tf       - f_star = %.2E" % (f-f_star))
                print("\tf_synch - f_star = %.2E" % (f_synch-f_star))

                plot_f.set(xdata=data_iteration, ydata=np.array(data_f)-f_star)
                plot_f_synch.set(xdata=data_iteration, ydata=np.array(data_f_synch)-f_star)
                ax1.set(xlim=(0,k), ylim=(ax1_y_min,ax1_y_max), title="F (asynch): %.2E\nF (synch): %.2E" % (f,f_synch))

                fig1.canvas.draw()
                fig1.canvas.flush_events()

    time_end = time.time()
    time_elapsed = time_end - time_start

    if show_fig:
        print("Finished simulation. Close figure to quit.")
        plt.show(block=True)

    # if save_output:
    #     file_path = os.path.join(save_directory_path, save_file_name)
    #     keys_descending_length = list(parameters.keys())
    #     keys_descending_length.sort(key=len, reverse=True)
    #     for i in range(len(keys_descending_length)):
    #         key = keys_descending_length[i]
    #         file_path = file_path.replace("<"+key+">", str(parameters[key]))
    #     os.makedirs(os.path.dirname(file_path), exist_ok=True)

    #     output_data = {"parameters":parameters, "iterations":data_iteration, "f":data_f, "f_synch":data_f_synch}
    #     with open(file_path, "w") as file:
    #         json.dump(output_data, file)
    # return time_elapsed

if __name__ == "__main__":
    parameters = {
        "rng_seed" : 0,
        "N" : 10,
        "B_comp1" : 2,
        "B_comp2" : 10,
        "B_comm" : 1000,
        "kmax" : 20000,
        "eta" : 0.3,
        "eta_synch" : 0.3,
        "alpha" : 0.02,
        "delta" : 1,
        "show_fig" : True,
    }
    from threadpoolctl import threadpool_limits
    with threadpool_limits(8, "blas"):
        run_sim(parameters)