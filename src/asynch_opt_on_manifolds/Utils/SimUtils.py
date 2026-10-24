import numpy as np
from asynch_opt_on_manifolds.Utils.Agents import Agent

def runSimCentralized(manifold, x_initial, eta, data, calculate_euclidean_grad, calculate_f, centralized_kmax, centralized_min_grad_norm):
    N = len(x_initial)

    # Create agents
    agents = []
    for i in range(N):
        agents.append(Agent(i, manifold, x_initial, eta, data, calculate_euclidean_grad))

    # Run sim
    # print("Solving centralized problem:")
    for k in range(centralized_kmax):
        # Compute
        for i in range(N):
            agents[i].compute()

        # Communicate
        for i in range(N):
            for j in range(N):
                if i==j:
                    continue
                agents[i].x[j] = agents[j].x[j]

        # Check gradient
        if k % 100 == 0:
            grad_norm_squared = 0
            for i in range(N):
                rie_grad = manifold.proj(agents[i].x[i], calculate_euclidean_grad(agents[i]))
                grad_norm_squared += np.linalg.norm(rie_grad[0])**2 + np.linalg.norm(rie_grad[1])**2
            grad_norm = np.sqrt(grad_norm_squared)

            # print("k=%g, gradient norm=%.2E   (stopping at k=%g, gradient norm=%.2E)" % (k, grad_norm, centralized_kmax, centralized_min_grad_norm))

            if grad_norm <= centralized_min_grad_norm:
                break


    # Compute final f
    x = []
    for i in range(N):
        x.append(agents[i].x[i])
    f = calculate_f(data, x)

    return f