import numpy as np
import matplotlib.pyplot as plt

from Manifolds import EuclideanManifold, UnitQuaternionManifold, ProductManifold
from RotationUtils import axis_angle_to_rot_mat, quat_to_rot_mat, partial_rot_mat_partial_quat


def copy(message):
    if type(message) == np.ndarray:
        message_copy = message.copy()
    elif type(message) == list:
        message_copy = []
        for i in range(len(message)):
            message_copy.append(copy(message[i]))
    else:
        raise ValueError("Attempted to copy unknown type (%s)." % str(type(message)))
    return message_copy


class Agent:
    def __init__(self, i, manifold, x_initial, lam, data):
        self.i = i
        self.manifold = manifold
        self.x = copy(x_initial)
        self.lam = lam

        self.alpha, self.prior_poses, self.formation_edges, self.formation_offsets = data

    def calculate_euclidean_grad(self):
        # Calculate gradient f wrt ti
        nabla_ti = 2* self.alpha * (self.x[self.i][0] - self.prior_poses[self.i][0])
        for k in range(len(self.formation_edges)):
            if self.i == self.formation_edges[k][0]:
                _,j = self.formation_edges[k]
                ttilde_ij = self.formation_offsets[k]
                Rqi = quat_to_rot_mat(self.x[self.i][1])
                nabla_ti += 2*(1-self.alpha)*(self.x[self.i][0] - self.x[j][0] + Rqi@ttilde_ij)
            elif self.i == self.formation_edges[k][1]:
                j,_ = self.formation_edges[k]
                ttilde_ji = self.formation_offsets[k]
                Rqj = quat_to_rot_mat(self.x[j][1])
                nabla_ti += 2*(1-self.alpha)*(self.x[self.i][0] - self.x[j][0] - Rqj@ttilde_ji)

        # Calculate gradient f wrt Rqi
        Rqi = quat_to_rot_mat(self.x[self.i][1])
        nabla_Rqi = 2 * self.alpha * (Rqi - self.prior_poses[self.i][1])
        for k in range(len(self.formation_edges)):
            if self.i == self.formation_edges[k][0]:
                j,_ = self.formation_edges[k]
                ttilde_ij = self.formation_offsets[k]
                nabla_Rqi += (1-self.alpha) * 2 * (self.x[j][0] - self.x[self.i][0]) @ (Rqi.T @ (self.x[j][0] - self.x[self.i][0]) - ttilde_ij).T

        # Calculate partial of Rqi wrt qic, c = [w,x,y,z]
        partials_Rqi = partial_rot_mat_partial_quat(self.x[self.i][1])

        # Calculate gradient of f wrt qi
        nabla_qi = np.zeros((4,1))
        for c in range(4):
            nabla_qi[c,0] = np.trace(nabla_Rqi.T @ partials_Rqi[c])

        nabla_xi = [nabla_ti, nabla_qi]
        return nabla_xi

    def compute(self):
        # Calculate Euclidean gradient
        euc_grad_xi = self.calculate_euclidean_grad()

        # Calculate Riemannian gradient via projection
        grad_xi = self.manifold.proj(self.x[self.i], euc_grad_xi)

        # Perform gradient step
        self.x[self.i] = self.manifold.exp(self.x[self.i], [-self.lam*grad_xi[0], -self.lam*grad_xi[1]])


class SynchAgent:
    def __init__(self, i, manifold, x_initial, lam, data):
        self.i = i
        self.manifold = manifold
        self.lam = lam

        self.alpha, self.prior_poses, self.formation_edges, self.formation_offsets = data

        self.current_iteration = 0
        self.current_x = copy(x_initial)
        self.next_x = [None for _ in range(len(x_initial))]
        self.sent_current = False

    def calculate_euclidean_grad(self):
        # Calculate gradient f wrt ti
        nabla_ti = 2* self.alpha * (self.current_x[self.i][0] - self.prior_poses[self.i][0])
        for k in range(len(self.formation_edges)):
            if self.i == self.formation_edges[k][0]:
                _,j = self.formation_edges[k]
                if self.current_x[j] is None:
                    return None
                ttilde_ij = self.formation_offsets[k]
                Rqi = quat_to_rot_mat(self.current_x[self.i][1])
                nabla_ti += 2*(1-self.alpha)*(self.current_x[self.i][0] - self.current_x[j][0] + Rqi@ttilde_ij)
            elif self.i == self.formation_edges[k][1]:
                j,_ = self.formation_edges[k]
                if self.current_x[j] is None:
                    return None
                ttilde_ji = self.formation_offsets[k]
                Rqj = quat_to_rot_mat(self.current_x[j][1])
                nabla_ti += 2*(1-self.alpha)*(self.current_x[self.i][0] - self.current_x[j][0] - Rqj@ttilde_ji)

        # Calculate gradient f wrt Rqi
        Rqi = quat_to_rot_mat(self.current_x[self.i][1])
        nabla_Rqi = 2 * self.alpha * (Rqi - self.prior_poses[self.i][1])
        for k in range(len(self.formation_edges)):
            if self.i == self.formation_edges[k][0]:
                _,j = self.formation_edges[k]
                if self.current_x[j] is None:
                    return None
                ttilde_ij = self.formation_offsets[k]
                nabla_Rqi += (1-self.alpha) * 2 * (self.current_x[j][0] - self.current_x[self.i][0]) @ (Rqi.T @ (self.current_x[j][0] - self.current_x[self.i][0]) - ttilde_ij).T

        # Calculate partial of Rqi wrt qic, c = [w,x,y,z]
        partials_Rqi = partial_rot_mat_partial_quat(self.current_x[self.i][1])

        # Calculate gradient of f wrt qi
        nabla_qi = np.zeros((4,1))
        for c in range(4):
            nabla_qi[c,0] = np.trace(nabla_Rqi.T @ partials_Rqi[c])

        nabla_xi = [nabla_ti, nabla_qi]
        return nabla_xi

    def compute(self):
        if self.sent_current: # Must send message before overwriting it
            # Calculate Euclidean gradient
            euc_grad_xi = self.calculate_euclidean_grad()

            if euc_grad_xi is not None:
                # Calculate Riemannian gradient via projection
                grad_xi = self.manifold.proj(self.current_x[self.i], euc_grad_xi)

                # Perform gradient step
                self.next_x[self.i] = self.manifold.exp(self.current_x[self.i], [-self.lam*grad_xi[0], -self.lam*grad_xi[1]])

                # Update buffers
                for i in range(len(self.next_x)):
                    if self.next_x[i] is None:
                        self.current_x[i] = None
                    else:
                        self.current_x[i] = copy(self.next_x[i])
                self.next_x = [None for _ in range(len(self.current_x))]
                self.current_iteration += 1
                self.sent_current = False


def main():
    # Define parameters
    rng_seed = 0
    N = 10
    B_comm = 1000
    B_compute = 100
    kmax = 10000
    lam = 0.0002
    lam_synch = 0.0002
    alpha = 0
    graph_edge_probability = 0.8
    render_iteration_mod = 100
    world_size = 10

    # Init RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Generate prior poses
    prior_poses = []
    for i in range(N):
        random_position = rng.uniform(low=np.zeros((3,1)), high=world_size*np.ones((3,1)), size=(3,1))
        random_quaternion = rng.multivariate_normal(mean=np.zeros(4), cov=np.eye(4)).reshape((4,1))
        random_unit_quaternion = random_quaternion / np.linalg.norm(random_quaternion)
        random_rot_mat = quat_to_rot_mat(random_unit_quaternion)
        prior_poses.append([random_position, random_rot_mat])

    # Generate formation graph
    adjacency_mat = None
    for attempt in range(20):
        A_rand = rng.uniform(size=(N,N))
        A_rand = (A_rand+A_rand.T)/2
        A = (A_rand <= graph_edge_probability) * 1
        # Check if connected
        D = np.diag(np.sum(A,axis=1))
        L = D - A
        connected = np.linalg.eigvalsh(L)[1] > 0
        if connected:
            adjacency_mat = A
            break
    if adjacency_mat is None:
        print("Failed to generate connected measurement graph in specified number of attempts.")
        return

    # Calculate formation edges
    formation_edges = []
    # for i in range(0,N-1):
    #     for j in range(i+1,N):
    for i in range(N):
        for j in range(N):
            if adjacency_mat[i,j] == 1:
                formation_edges.append((i,j))

    # Generate formation positions
    formation_positions = []
    for i in range(N):
        position = rng.uniform(low=np.zeros((3,1)), high=world_size*np.ones((3,1)), size=(3,1))
        formation_positions.append(position)

    # Generate formation offsets
    formation_offsets = []
    for k in range(len(formation_edges)):
        i,j = formation_edges[k]
        offset = formation_positions[j] - formation_positions[i]
        formation_offsets.append(offset)

    # Generate initial solution value
    x_initial = []
    for i in range(N):
        random_position = rng.uniform(low=np.zeros((3,1)), high=world_size*np.ones((3,1)), size=(3,1))
        random_quaternion = rng.multivariate_normal(mean=np.zeros(4), cov=np.eye(4)).reshape((4,1))
        random_unit_quaternion = random_quaternion / np.linalg.norm(random_quaternion)
        xi = [random_position, random_unit_quaternion]
        x_initial.append(xi)

    # Initialize agents
    agents = []
    agents_synch = []
    data = (alpha, prior_poses, formation_edges, formation_offsets)
    for i in range(N):
        agents.append(Agent(i, ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, lam, data))
        agents_synch.append(SynchAgent(i, ProductManifold([EuclideanManifold(), UnitQuaternionManifold()]), x_initial, lam_synch, data))

    # Initialize mailboxes
    mailbox_outgoing = []
    for i in range(N):
        mailbox_outgoing.append(copy(x_initial[i]))
    mailbox_synch_incoming = [[] for i in range(N)]

    # Initialize iterations at which agents compute, send, and receive
    iteration_to_comp = []
    iteration_to_send = []
    iteration_to_recv = []
    for i in range(N):
        iteration_to_comp.append(rng.integers(low=1,high=B_compute+1))
        iteration_to_send.append(rng.integers(low=1,high=B_comm//2+1))
        iteration_to_recv.append(rng.integers(low=1,high=B_comm//2+1))

    # Init plot data
    data_iteration = []
    data_f = []
    data_f_synch = []
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
                iteration_to_send[i] = k + rng.integers(low=1,high=B_comm//2+1)
                # Let asynch agents send
                mailbox_outgoing[i] = copy(agents[i].x[i])
                # Let synch agents send
                if not agents_synch[i].sent_current:
                    agents_synch[i].sent_current = True
                    for j in range(N):
                        if j != i:
                            message = (i, agents_synch[i].current_iteration, copy(agents_synch[i].current_x[i]))
                            mailbox_synch_incoming[j].append(message)

        # Recieve
        for i in range(N):
            if iteration_to_recv[i] == k:
                iteration_to_recv[i] = k + rng.integers(low=1,high=B_comm//2+1)
                # Let asynch agents receive
                for j in range(N):
                    if i != j:
                        agents[i].x[j] = copy(mailbox_outgoing[j])
                # Let synch agents receive
                for message in mailbox_synch_incoming[i]:
                    j, message_iteration, message_xj = message
                    if message_iteration < agents_synch[i].current_iteration:
                        pass
                    elif message_iteration == agents_synch[i].current_iteration:
                        agents_synch[i].current_x[j] = copy(message_xj)
                    elif message_iteration == agents_synch[i].current_iteration + 1:
                        agents_synch[i].next_x[j] = copy(message_xj)
                    else:
                        pass
                        # print("Error: Agent %g at iteration %g received a message from agent %g, timestamped at iteration %g." % (i, agents_synch[i].current_iteration, j, message_iteration))
                        # return
                mailbox_synch_incoming[i] = []

        # Compute
        for i in range(N):
            if iteration_to_comp[i] == k:
                iteration_to_comp[i] = k + rng.integers(low=1,high=B_compute+1)
                agents[i].compute()
                agents_synch[i].compute()

        # Generate analytical asynch true x
        x = []
        for i in range(N):
            x.append(copy(agents[i].x[i]))

        # Update analytical synch true x map
        min_x_synch_iteration = 1e10
        for i in range(N):
            synch_iteration_i = agents_synch[i].current_iteration
            synch_xi = copy(agents_synch[i].current_x[i])
            min_x_synch_iteration = min(min_x_synch_iteration, synch_iteration_i)
            if synch_iteration_i not in x_synch_map:
                x_synch_map[synch_iteration_i] = [None for i in range(N)]
            x_synch_map[synch_iteration_i][i] = synch_xi

        # Remove outdated synch x from map
        for map_k in list(x_synch_map.keys()):
            if map_k < min_x_synch_iteration:
                del x_synch_map[map_k]

        x_synch = x_synch_map[min_x_synch_iteration]

        iteration_synch = min_x_synch_iteration

        # Calculate f asynch and f synch
        f = 0
        f_synch = 0
        for i in range(N):
            f += alpha * np.linalg.norm(x[i][0] - prior_poses[i][0])**2 + alpha * np.linalg.norm(quat_to_rot_mat(x[i][1]) - prior_poses[i][1], "fro")**2
            f_synch += alpha * np.linalg.norm(x_synch[i][0] - prior_poses[i][0])**2 + alpha * np.linalg.norm(quat_to_rot_mat(x_synch[i][1]) - prior_poses[i][1], "fro")**2
        for e in range(len(formation_edges)):
            i,j = formation_edges[e]
            ttilde_ij = formation_offsets[e]
            f += (1 - alpha) * np.linalg.norm(quat_to_rot_mat(x[i][1]).T @ (x[j][0] - x[i][0]) - ttilde_ij)**2
            f_synch += (1 - alpha) * np.linalg.norm(quat_to_rot_mat(x_synch[i][1]).T @ (x_synch[j][0] - x_synch[i][0]) - ttilde_ij)**2

        # Update plots
        data_iteration.append(k)
        data_f.append(f)
        data_f_synch.append(f_synch)
        ax1_y_min = min(ax1_y_min, min(f, f_synch))
        ax1_y_max = max(ax1_y_max, max(f, f_synch))

        if k % render_iteration_mod == 0 and ax1_y_min < ax1_y_max:
            print("Iteration %4g: f = %.4E" % (k, f))
            print("\tSynch Iteration %g: f_synch = %.4E" % (iteration_synch, f_synch))
            print("\tSynch Iterations: %s" % (str([agents_synch[i].current_iteration for i in range(N)])))

            plot_f.set(xdata=data_iteration, ydata=data_f)
            plot_f_synch.set(xdata=data_iteration, ydata=data_f_synch)
            ax1.set(xlim=(0,k), ylim=(ax1_y_min,ax1_y_max), title="F (asynch): %.2E\nF (synch): %.2E" % (f,f_synch))

            fig.canvas.draw()
            fig.canvas.flush_events()

    print("Finished simulation. Close figure to quit.")
    plt.show(block=True)


if __name__ == "__main__":
    main()