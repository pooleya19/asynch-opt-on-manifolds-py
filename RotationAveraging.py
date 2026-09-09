import numpy as np
import matplotlib.pyplot as plt

from Manifolds import UnitQuaternionManifold
from RotationUtils import axis_angle_to_rot_mat, quat_to_rot_mat, partial_rot_mat_partial_quat


class Agent:
    def __init__(self, i, manifold, x_initial, lam, edges, measurements):
        self.i = i
        self.manifold = manifold
        self.x = [x_initial[i].copy() for i in range(len(x_initial))]
        self.lam = lam
        self.edges = edges
        self.measurements = measurements

    def calculate_euclidean_grad(self):
        # Calculate gradient of f wrt R(x_i)
        nabla_R_x_i = np.zeros((3,3))
        for k in range(len(self.edges)):
            if self.i == self.edges[k][0]:
                i,j = self.edges[k]
                Rtilde_ij = self.measurements[k]

                R_x_i = quat_to_rot_mat(self.x[i])
                R_x_j = quat_to_rot_mat(self.x[j])

                nabla_R_x_i += 2*R_x_i - 2*R_x_j @ Rtilde_ij.T
                
            elif self.i == self.edges[k][1]:
                j,i = self.edges[k]
                Rtilde_ji = self.measurements[k]

                R_x_i = quat_to_rot_mat(self.x[i])
                R_x_j = quat_to_rot_mat(self.x[j])

                nabla_R_x_i += 2*R_x_i - 2*R_x_j @ Rtilde_ji

        # Calculate partial of R(x_i) wrt x_ic, c = [w,x,y,z]
        partials_R_x_i = partial_rot_mat_partial_quat(self.x[self.i])

        # Calculate gradient of f wrt x_i
        nabla_x_i = np.zeros((4,1))
        for i in range(4):
            nabla_x_i[i,0] = np.trace(nabla_R_x_i.T @ partials_R_x_i[i])
        return nabla_x_i

    def compute(self):
        # Calculate Euclidean gradient
        euc_grad_xi = self.calculate_euclidean_grad()

        # Calculate Riemannian gradient via projection
        grad_xi = self.manifold.proj(self.x[self.i], euc_grad_xi)

        # Perform gradient step
        self.x[self.i] = self.manifold.exp(self.x[self.i], -self.lam*grad_xi)


class SynchAgent:
    def __init__(self, i, manifold, x_initial, lam, edges, measurements):
        self.i = i
        self.manifold = manifold
        self.lam = lam
        self.edges = edges
        self.measurements = measurements

        self.current_iteration = 0
        self.current_x = [x_initial[i].copy() for i in range(len(x_initial))]
        self.next_x = [None for i in range(len(x_initial))]
        self.sent_current = False

    def calculate_euclidean_grad(self):
        # Calculate gradient of f wrt R(x_i)
        nabla_R_x_i = np.zeros((3,3))
        for k in range(len(self.edges)):
            if self.i == self.edges[k][0]:
                i,j = self.edges[k]
                Rtilde_ij = self.measurements[k]

                if self.current_x[j] is None:
                    return None
                R_x_i = quat_to_rot_mat(self.current_x[i])
                R_x_j = quat_to_rot_mat(self.current_x[j])

                nabla_R_x_i += 2*R_x_i - 2*R_x_j @ Rtilde_ij.T
                
            elif self.i == self.edges[k][1]:
                j,i = self.edges[k]
                Rtilde_ji = self.measurements[k]

                if self.current_x[j] is None:
                    return None
                R_x_i = quat_to_rot_mat(self.current_x[i])
                R_x_j = quat_to_rot_mat(self.current_x[j])

                nabla_R_x_i += 2*R_x_i - 2*R_x_j @ Rtilde_ji

        # Calculate partial of R(x_i) wrt x_ic, c = [w,x,y,z]
        partials_R_x_i = partial_rot_mat_partial_quat(self.current_x[self.i])

        # Calculate gradient of f wrt x_i
        nabla_x_i = np.zeros((4,1))
        for i in range(4):
            nabla_x_i[i,0] = np.trace(nabla_R_x_i.T @ partials_R_x_i[i])
        return nabla_x_i

    def compute(self):
        if self.sent_current: # Must send message before overwriting it
            # Calculate Euclidean gradient
            euc_grad_xi = self.calculate_euclidean_grad()

            if euc_grad_xi is not None:
                # Calculate Riemannian gradient via projection
                grad_xi = self.manifold.proj(self.current_x[self.i], euc_grad_xi)

                # Perform gradient step
                self.next_x[self.i] = self.manifold.exp(self.current_x[self.i], -self.lam*grad_xi)

                # Update buffers
                for i in range(len(self.next_x)):
                    if self.next_x[i] is None:
                        self.current_x[i] = None
                    else:
                        self.current_x[i] = self.next_x[i].copy()
                self.next_x = [None for i in range(len(self.current_x))]
                self.current_iteration += 1
                self.sent_current = False



def main():
    # Define parameters
    rng_seed = 0
    N = 10
    B_comm = 1000
    B_compute = 50
    kmax = 10000
    lam = 0.01
    lam_synch = 0.01
    measurement_probability = 0.5
    measurement_noise_angle_std_dev = 0.5
    render_iteration_mod = 100

    # Init RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Generate true orientations
    x_true = []
    for i in range(N):
        xi_true_rand = rng.multivariate_normal(mean=np.zeros(4), cov=np.eye(4)).reshape((4,1))
        xi_true = xi_true_rand / np.linalg.norm(xi_true_rand)
        x_true.append(xi_true)

    # Generate measurement graph
    adjacency_mat = None
    for attempt in range(20):
        A_rand = rng.uniform(size=(N,N))
        A_rand = (A_rand+A_rand.T)/2
        A = (A_rand <= measurement_probability) * 1
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

    # Generate noisy measurements
    edges = []
    measurements = []
    for i in range(0,N-1):
        for j in range(i+1,N):
            if adjacency_mat[i,j] == 1:
                axis_rand_rand = rng.multivariate_normal(mean=np.zeros(3), cov=np.eye(3)).reshape((3,1))
                axis_rand = axis_rand_rand / np.linalg.norm(axis_rand_rand)
                angle_rand = rng.normal(loc=0, scale=measurement_noise_angle_std_dev)
                V_ij = axis_angle_to_rot_mat(axis_rand, angle_rand)

                R_x_i = quat_to_rot_mat(x_true[i])
                R_x_j = quat_to_rot_mat(x_true[j])

                Rtilde_ij = R_x_i.T @ R_x_j @ V_ij
                edges.append((i,j))
                measurements.append(Rtilde_ij)

    # Generate initial solution value
    x_initial = []
    quat_ident = np.vstack((1, np.zeros((3,1))))
    for i in range(N):
        x_initial.append(quat_ident)

    # Initialize agents
    agents = []
    agents_synch = []
    for i in range(N):
        agents.append(Agent(i, UnitQuaternionManifold(), x_initial, lam, edges, measurements))
        agents_synch.append(SynchAgent(i, UnitQuaternionManifold(), x_initial, lam_synch, edges, measurements))

    # Initialize mailboxes
    mailbox_outgoing = []
    for i in range(N):
        mailbox_outgoing.append(x_initial[i].copy())
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

    # Run simulation
    for k in range(kmax):
        # Send
        for i in range(N):
            if iteration_to_send[i] == k:
                iteration_to_send[i] = k + rng.integers(low=1,high=B_comm//2+1)
                # Let asynch agents send
                mailbox_outgoing[i] = agents[i].x[i].copy()
                # Let synch agents send
                if not agents_synch[i].sent_current:
                    agents_synch[i].sent_current = True
                    for j in range(N):
                        if j != i:
                            message = (i, agents_synch[i].current_iteration, agents_synch[i].current_x[i].copy())
                            mailbox_synch_incoming[j].append(message)

        # Recieve
        for i in range(N):
            if iteration_to_recv[i] == k:
                iteration_to_recv[i] = k + rng.integers(low=1,high=B_comm//2+1)
                # Let asynch agents receive
                for j in range(N):
                    if i != j:
                        agents[i].x[j] = mailbox_outgoing[j].copy()
                # Let synch agents receive
                for message in mailbox_synch_incoming[i]:
                    j, message_iteration, message_xj = message
                    if message_iteration < agents_synch[i].current_iteration:
                        pass
                    elif message_iteration == agents_synch[i].current_iteration:
                        agents_synch[i].current_x[j] = message_xj.copy()
                    elif message_iteration == agents_synch[i].current_iteration + 1:
                        agents_synch[i].next_x[j] = message_xj.copy()
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
            x.append(agents[i].x[i].copy())

        # Generate analytical synch true x
        x_synch = []
        for i in range(N):
            x_synch.append(agents_synch[i].current_x[i].copy())

        iteration_synch = np.min([agents_synch[i].current_iteration for i in range(N)])

        # Calculate f asynch and f synch
        f = 0
        f_synch = 0
        for edge_index in range(len(edges)):
            i,j = edges[edge_index]
            Rtilde_ij = measurements[edge_index]
            f += np.linalg.norm(quat_to_rot_mat(x[i]).T @ quat_to_rot_mat(x[j]) - Rtilde_ij, "fro")**2
            f_synch += np.linalg.norm(quat_to_rot_mat(x_synch[i]).T @ quat_to_rot_mat(x_synch[j]) - Rtilde_ij, "fro")**2

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