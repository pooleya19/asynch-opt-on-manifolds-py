import numpy as np
from asynch_opt_on_manifolds.Utils.ManifoldUtils import scale_tangent_vector, freeze_manifold_point


class Agent:
    def __init__(self, i, manifold, x_initial, eta, data, calculate_euclidean_grad):
        self.i = i
        self.manifold = manifold
        self.x = list(x_initial)
        self.eta = eta
        self.data = data
        self.calculate_euclidean_grad = calculate_euclidean_grad

    def compute(self):
        # Calculate Euclidean gradient
        euc_grad_xi = self.calculate_euclidean_grad(self)
        assert euc_grad_xi is not None, "Agent %g's euclidean gradient is None." % self.i

        # Calculate Riemannian gradient via projection
        grad_xi = self.manifold.proj(self.x[self.i], euc_grad_xi)

        # Perform gradient step
        scaled_grad_xi = scale_tangent_vector(grad_xi, -self.eta)
        self.x[self.i] = freeze_manifold_point(self.manifold.exp(self.x[self.i], scaled_grad_xi))


class SynchAgent:
    def __init__(self, i, manifold, x_initial, eta, data, calculate_euclidean_grad):
        self.i = i
        self.manifold = manifold
        self.eta = eta
        self.data = data
        self.calculate_euclidean_grad = calculate_euclidean_grad

        self.x = list(x_initial)
        self.next_x = [None for _ in range(len(self.x))]
        self.current_iteration = 0
        self.sent_current = False

    def compute(self):
        if self.sent_current: # Must send message before overwriting it
            # Calculate Euclidean gradient
            euc_grad_xi = self.calculate_euclidean_grad(self)

            if euc_grad_xi is not None:
                # Calculate Riemannian gradient via projection
                grad_xi = self.manifold.proj(self.x[self.i], euc_grad_xi)

                # Perform gradient step
                scaled_grad_xi = scale_tangent_vector(grad_xi, -self.eta)
                self.next_x[self.i] = freeze_manifold_point(self.manifold.exp(self.x[self.i], scaled_grad_xi))

                # Update buffers
                self.x = self.next_x
                self.next_x = [None for _ in range(len(self.x))]
                self.current_iteration += 1
                self.sent_current = False
                return True
        return False