import numpy as np
import matplotlib.pyplot as plt

from Manifolds import EuclideanManifold


class Agent:
    def __init__(self, i, manifold, x_initial, lam, edges, measurements):
        self.i = i
        self.manifold = manifold
        self.x = [x_initial[i].copy() for i in range(len(x_initial))]
        self.lam = lam
        self.edges = edges
        self.measurements = measurements

    def calculate_euclidean_grad(self):
        euc_grad_xi = np.zeros((3,1))
        for k in range(len(self.edges)):
            if self.i == self.edges[k][0]:
                j = self.edges[k][1]
                dtilde_ij = self.measurements[k][0]
                R_ij = self.measurements[k][1]
                euc_grad_xi -= 2*np.linalg.inv(R_ij) @ (self.x[j] - self.x[self.i] - dtilde_ij)
                
            elif self.i == self.edges[k][1]:
                j = self.edges[k][0]
                dtilde_ji = self.measurements[k][0]
                R_ji = self.measurements[k][1]
                euc_grad_xi += 2*np.linalg.inv(R_ji) @ (self.x[self.i] - self.x[j] - dtilde_ji)
        return euc_grad_xi

    def compute(self):
        # Calculate Euclidean gradient
        euc_grad_xi = self.calculate_euclidean_grad()

        # Calculate Riemannian gradient via projection
        grad_xi = self.manifold.proj(self.x[self.i], euc_grad_xi)

        # Perform gradient step
        self.x[self.i] = self.manifold.exp(self.x[self.i], -self.lam*grad_xi)


def main():
    # Define parameters
    rng_seed = 0
    N = 10
    B = 100
    kmax = 2000
    lam = 0.00001
    world_size = 10 
    measurement_probability = 0.5
    measurement_std_dev = 0.01
    render_iteration_mod = 50

    # Init RNG
    rng = np.random.default_rng(seed=rng_seed)

    # Generate true positions
    x_true = []
    for i in range(N):
        # Assume world is cube in positive orthant, with a vertex at (0,0,0)
        xi_true = rng.uniform(low=np.zeros((3,1)), high=world_size*np.ones((3,1)), size=(3,1))
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
                R_ij = measurement_std_dev**2 * np.eye(3)
                noise_ij = rng.multivariate_normal(mean=np.zeros(3), cov=R_ij).reshape((3,1))
                dtilde_ij = x_true[j] - x_true[i] + noise_ij
                edges.append((i,j))
                measurements.append((dtilde_ij, R_ij))

    # Generate initial solution value
    x_initial = []
    for i in range(N):
        x_initial.append(np.zeros((3,1)))

    # Initialize agents
    agents = []
    for i in range(N):
        agents.append(Agent(i, EuclideanManifold(), x_initial, lam, edges, measurements))

    # Initialize outgoing mailbox
    mailbox_outgoing = []
    for i in range(N):
        mailbox_outgoing.append(x_initial[i].copy())

    # Initialize iterations at which agents compute, send, and receive
    iteration_to_comp = []
    iteration_to_send = []
    iteration_to_recv = []
    for i in range(N):
        iteration_to_comp.append(rng.integers(low=1,high=B+1))
        iteration_to_send.append(rng.integers(low=1,high=B//2+1))
        iteration_to_recv.append(rng.integers(low=1,high=B//2+1))

    # Init plot data
    data_iteration = []
    data_obj_func = []
    data_obj_func_min = 1e10
    data_obj_func_max = 0
    data_mse = []
    data_mse_min = 1e10
    data_mse_max = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    plot_f = ax1.plot([], [], "-r")[0]
    ax1.set(xlabel="Iteration", ylabel="Objective Function", yscale="log", title="Objective Function: %.2E"%0)

    plot_mse = ax2.plot([], [], "-b")[0]
    ax1.set(xlabel="Iteration", ylabel="MSE Function", yscale="log", title="MSE: %.2E"%0)

    plt.show(block=False)

    # Run simulation
    for k in range(kmax):
        # Send
        for i in range(N):
            if iteration_to_send[i] == k:
                iteration_to_send[i] = k + rng.integers(low=1,high=B//2+1)
                mailbox_outgoing[i] = agents[i].x[i].copy()

        # Recieve
        for i in range(N):
            if iteration_to_recv[i] == k:
                iteration_to_recv[i] = k + rng.integers(low=1,high=B//2+1)
                for j in range(N):
                    if i != j:
                        agents[i].x[j] = mailbox_outgoing[j].copy()

        # Compute
        for i in range(N):
            if iteration_to_comp[i] == k:
                iteration_to_comp[i] = k + rng.integers(low=1,high=B+1)
                agents[i].compute()

        # Generate analytical true x
        x = []
        for i in range(N):
            x.append(agents[i].x[i].copy())

        # Calculate f
        f = 0
        for edge_index in range(len(edges)):
            i = edges[edge_index][0]
            j = edges[edge_index][1]
            dtilde_ij = measurements[edge_index][0]
            R_ij = measurements[edge_index][1]
            f += (x[j] - x[i] - dtilde_ij).T @ np.linalg.inv(R_ij) @ (x[j] - x[i] - dtilde_ij)
        f = f.item()

        # Calculate MSE
        mse = 0
        for edge in edges:
            i,j = edge
            mse += 1/len(edges) * np.linalg.norm(x[j]-x[i] - (x_true[j]-x_true[i]))**2

        # Update plots
        data_iteration.append(k)
        data_obj_func.append(f)
        data_obj_func_min = min(data_obj_func_min, f)
        data_obj_func_max = max(data_obj_func_max, f)
        data_mse.append(mse)
        data_mse_min = min(data_mse_min, mse)
        data_mse_max = max(data_mse_max, mse)

        if k % render_iteration_mod == 0 and data_obj_func_min < data_obj_func_max and data_mse_min < data_mse_max:
            print("Iteration %4g: f = %.4E" % (k, f))
            print("\tMSE: %.4E" % mse)
            # print("\tx:")
            # for i in range(N):
            #     print("\t\t%s" % (str(x[i].T)))

            plot_f.set(xdata=data_iteration, ydata=data_obj_func)
            ax1.set(xlim=(0,k), ylim=(data_obj_func_min,1.2*data_obj_func_max), title="Objective Function: %.2E"%f)

            plot_mse.set(xdata=data_iteration, ydata=data_mse)
            ax2.set(xlim=(0,k), ylim=(data_mse_min,1.2*data_mse_max), title="MSE: %.2E"%mse)

            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.show(block=True)


if __name__ == "__main__":
    main()