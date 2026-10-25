import numpy as np

from asynch_opt_on_manifolds.Utils.Agents import Agent
from asynch_opt_on_manifolds.Utils.Manifolds import ProductManifold
from asynch_opt_on_manifolds.Utils.ManifoldUtils import scale_tangent_vector, sum_tangent_vectors


def runSimCentralizedRGD(manifold, x_initial, eta, data, calculate_euclidean_grad, calculate_f, centralized_kmax, centralized_min_grad_norm):
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


# Implemented from https://doi.org/10.1093/imanum/drx015
def runSimCentralizedRBB(manifold, x_initial, eta, data, calculate_euclidean_grad, calculate_f, centralized_min_grad_norm):
    sigma = 0.5
    gamma = 1e-3
    M = 100
    alpha_BB_min = 1e-8
    alpha_BB_max = 1e8
    alpha_BB = eta

    N = len(x_initial)
    product_manifold = ProductManifold([manifold for _ in range(N)])

    def rie_grad(x):
        return product_manifold.proj(x, [calculate_euclidean_grad(Agent(i, manifold, x, None, data, None)) for i in range(N)])

    # Step 1
    f0 = calculate_f(data, x_initial)
    recent_f = [f0]
    x = x_initial
    g = rie_grad(x)
    norm_g = product_manifold.norm(g)

    k = -1
    while True:
        k += 1
        if k%10 == 0:
            print("k=%5g, f=%.20f, gradient norm=%.2E   (stopping when gradient norm <= %.2E)" % (k, recent_f[0], norm_g, centralized_min_grad_norm))

        # Step 2
        h = 0
        while True:
            assert h < 1000, "h got too big."
            left = calculate_f(data, product_manifold.exp(x, scale_tangent_vector(g, -sigma**h * alpha_BB)))
            right = np.max(recent_f) - gamma * sigma**h * alpha_BB * norm_g**2
            if left <= right:
                break
            else:
                h += 1
        alpha = sigma**h * alpha_BB

        # Step 3
        x_old = x
        g_old = g
        x = product_manifold.exp(x, scale_tangent_vector(g, -alpha))
        g = rie_grad(x)
        norm_g = product_manifold.norm(g)
        if norm_g <= centralized_min_grad_norm:
            break
        
        f = calculate_f(data, x)
        recent_f.append(f)
        recent_f = recent_f[len(recent_f)-M : len(recent_f)] # Keep most recent M

        # Step 4
        Tg_old = product_manifold.transport(x_old, x, g_old)
        s = scale_tangent_vector(Tg_old, -alpha)
        y = sum_tangent_vectors(g, scale_tangent_vector(Tg_old, -1))

        # Step 5
        tau = product_manifold.inner_product(s,s) / product_manifold.inner_product(s,y)

        # Step 6
        if tau > 0:
            alpha_BB = min(alpha_BB_max, max(alpha_BB_min, tau))
        else:
            alpha_BB = alpha_BB_max

    return x, calculate_f(data, x)