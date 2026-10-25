import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import json
import time
import os
import cv2
from pathlib import Path


from asynch_opt_on_manifolds.Utils.Manifolds import EuclideanManifold, UnitQuaternionManifold, ProductManifold
from asynch_opt_on_manifolds.Utils.ManifoldUtils import freeze_manifold_point
from asynch_opt_on_manifolds.Utils.Agents import Agent, SynchAgent
from asynch_opt_on_manifolds.Utils.RotationUtils import quat_to_rot_mat, partial_rot_mat_partial_quat
from asynch_opt_on_manifolds.RobustFormationControl.Formations import generateFormationLattice
from asynch_opt_on_manifolds.Utils.SimUtils import runSimCentralizedRGD, runSimCentralizedRBB


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


def formation_geometry(x, formation_edges, s=0.1):
    P = np.hstack([xi[0] for xi in x]).T                    # (N,3) positions
    R = np.stack([quat_to_rot_mat(xi[1]) for xi in x])      # (N,3,3)
    E = np.array(formation_edges)                           # (n_edges,2)
    edge_segs = np.stack([P[E[:,0]], P[E[:,1]]], axis=1)    # (n_edges,2,3)
    axis_segs = [np.stack([P, P + s*R[:,:,c]], axis=1)      # 3 x (N,2,3)
                 for c in range(3)]
    return P, edge_segs, axis_segs


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
    save_final_fig = parameters["save_final_fig"]
    save_final_fig_name = parameters["save_final_fig_name"]
    save_video = parameters["save_video"]
    save_video_name = parameters["save_video_name"]
    save_data = parameters["save_data"]
    save_data_name = parameters["save_data_name"]

    # Define parameters
    render_fig = show_fig or save_video or save_final_fig
    compute_metrics_mod = 10
    render_mod = 100

    # Init RNG
    rng_static = np.random.default_rng(seed=0)
    rng_sim = np.random.default_rng(seed=rng_seed)

    # Generate prior poses
    prior_poses = []
    for i in range(N):
        random_position = rng_static.uniform(low=-0.5*np.ones((3,1)), high=0.5*np.ones((3,1)), size=(3,1))
        random_quaternion = rng_static.multivariate_normal(mean=np.zeros(4), cov=np.eye(4)).reshape((4,1))
        random_unit_quaternion = random_quaternion / np.linalg.norm(random_quaternion)
        prior_poses.append([random_position, random_unit_quaternion])

    # Generate formation
    formation_positions, formation_edges, formation_offsets = generateFormationLattice(N)
    x_formation = [[formation_positions[i], np.reshape([1,0,0,0], (4,1))] for i in range(N)]
    data = (alpha, delta, prior_poses, formation_edges, formation_offsets)

    # Initialize solution
    x_initial = list(prior_poses)
    freeze_manifold_point(x_initial)

    # Solve centralized problem
    centralized_min_grad_norm = 1e-12
    # print("Solving centralized problem...")
    # f_star = runSimCentralizedRGD(ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, eta_synch, data, calculate_euclidean_grad, calculate_f, 100000, centralized_min_grad_norm)
    f_star = runSimCentralizedRBB(ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, eta_synch, data, calculate_euclidean_grad, calculate_f, centralized_min_grad_norm)
    # f_star = 1.88e-2
    # print("Finished solving centralized problem! f_star = %.2E" % f_star)

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
    if render_fig:
        fig1, (ax1) = plt.subplots(1, 1, figsize=(8,5))

        plot_f = ax1.plot([], [], "-r", label="f (asynch)")[0]
        plot_f_synch = ax1.plot([], [], "-b", label="f (synch)")[0]
        ax1.set(xlabel="Iteration", ylabel="f - fstar", yscale="log", title="F (asynch): %.2E\nF (synch): %.2E" % (0,0))
        ax1.legend(loc="lower left")

        fig2, ax_form = plt.subplots(2, 2, figsize=(10,10), subplot_kw={"projection": "3d"})
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
        
        positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_initial, formation_edges, axes_length)
        ax_form_asynch_positions = ax_form_asynch.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3)[0]
        ax_form_asynch_axes_x = Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth)
        ax_form_asynch_axes_y = Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth)
        ax_form_asynch_axes_z = Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth)
        ax_form_asynch_edges = Line3DCollection(edges, colors="gray", linewidths=edge_linewidth)
        ax_form_asynch.add_collection3d(ax_form_asynch_axes_x)
        ax_form_asynch.add_collection3d(ax_form_asynch_axes_y)
        ax_form_asynch.add_collection3d(ax_form_asynch_axes_z)
        ax_form_asynch.add_collection3d(ax_form_asynch_edges)
        ax_form_asynch.set(title="Poses (Asynch)", xlim=(-lim,lim), ylim=(-lim,lim), zlim=(-lim,lim), aspect="equal")
        
        positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_initial, formation_edges, axes_length)
        ax_form_synch_positions = ax_form_synch.plot(positions[:,0], positions[:,1], positions[:,2], "o", color="k", markersize=3)[0]
        ax_form_synch_axes_x = Line3DCollection(axes_x, colors="r", linewidths=axes_linewidth)
        ax_form_synch_axes_y = Line3DCollection(axes_y, colors="g", linewidths=axes_linewidth)
        ax_form_synch_axes_z = Line3DCollection(axes_z, colors="b", linewidths=axes_linewidth)
        ax_form_synch_edges = Line3DCollection(edges, colors="gray", linewidths=edge_linewidth)
        ax_form_synch.add_collection3d(ax_form_synch_axes_x)
        ax_form_synch.add_collection3d(ax_form_synch_axes_y)
        ax_form_synch.add_collection3d(ax_form_synch_axes_z)
        ax_form_synch.add_collection3d(ax_form_synch_edges)
        ax_form_synch.set(title="Poses (Synch)", xlim=(-lim,lim), ylim=(-lim,lim), zlim=(-lim,lim), aspect="equal")

        fig1.canvas.draw()
        fig1.canvas.flush_events()
        fig2.canvas.draw()
        fig2.canvas.flush_events()

        if show_fig:
            plt.show(block=False)
            plt.pause(0.01)

    # Init video
    if save_video:
        fi2_width, fig2_height = fig2.canvas.get_width_height(physical=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        os.makedirs(os.path.dirname(save_video_name), exist_ok=True)
        video = cv2.VideoWriter(save_video_name, fourcc, 30.0, (fi2_width, fig2_height))


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
        if render_fig and k % render_mod == 0:
            if ax1_y_min < ax1_y_max:
                print("Iteration %g, Synch Iteration %g:" % (k, iteration_synch))
                print("\tf       - f_star = %.2E" % (f-f_star))
                print("\tf_synch - f_star = %.2E" % (f_synch-f_star))

                plot_f.set(xdata=data_iteration, ydata=np.array(data_f)-f_star)
                plot_f_synch.set(xdata=data_iteration, ydata=np.array(data_f_synch)-f_star)
                ax1.set(xlim=(0,k), ylim=(ax1_y_min,ax1_y_max), title="F (asynch): %.2E\nF (synch): %.2E" % (f,f_synch))

                positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x, formation_edges, axes_length)
                ax_form_asynch_positions.set_data_3d(positions[:,0], positions[:,1], positions[:,2])
                ax_form_asynch_axes_x.set_segments(axes_x)
                ax_form_asynch_axes_y.set_segments(axes_y)
                ax_form_asynch_axes_z.set_segments(axes_z)
                ax_form_asynch_edges.set_segments(edges)

                positions, axes_x, axes_y, axes_z, edges = generate_formation_visuals(x_synch, formation_edges, axes_length)
                ax_form_synch_positions.set_data_3d(positions[:,0], positions[:,1], positions[:,2])
                ax_form_synch_axes_x.set_segments(axes_x)
                ax_form_synch_axes_y.set_segments(axes_y)
                ax_form_synch_axes_z.set_segments(axes_z)
                ax_form_synch_edges.set_segments(edges)

                fig1.canvas.draw()
                fig1.canvas.flush_events()
                fig2.canvas.draw()
                fig2.canvas.flush_events()

                if save_video:
                    img = np.asarray(fig2.canvas.buffer_rgba())[:,:,:3]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img_bgr)


    time_end = time.time()
    time_elapsed = time_end - time_start

    if save_final_fig:
        os.makedirs(os.path.dirname(save_final_fig_name), exist_ok=True)
        fig2.savefig(save_final_fig_name, format="pdf")

    if save_video:
        video.release()

    if save_data:
        os.makedirs(os.path.dirname(save_data_name), exist_ok=True)

        output_data = {"parameters" : parameters, 
                       "iterations" : data_iteration, 
                       "f" : data_f, 
                       "f_synch" : data_f_synch, 
                       "f_star" : f_star,
                       "synch_iterations" : iteration_synch,
                       "time_elapsed" : time_elapsed}
        
        with open(save_data_name, "w") as file:
            json.dump(output_data, file)

    if show_fig:
        print("Finished simulation. Close figure to quit.")
        plt.show(block=True)

    return time_elapsed


if __name__ == "__main__":
    default_parameters = {
        "rng_seed" : 0,
        "N" : 32,
        "B_comp1" : 2,
        "B_comp2" : 10,
        "B_comm" : 1000,
        "kmax" : 10000,
        "eta" : 0.2,
        "eta_synch" : 0.2,
        "alpha" : 1e-4,
        "delta" : 1,
        "show_fig" : True,
        "save_final_fig" : False,
        "save_final_fig_name" : None,
        "save_video" : False,
        "save_video_name" : None,
        "save_data" : False,
        "save_data_name" : None,
    }

    from threadpoolctl import threadpool_limits
    with threadpool_limits(8, "blas"):
        run_sim(default_parameters)