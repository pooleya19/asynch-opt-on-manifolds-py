import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    @abstractmethod
    def exp(self, x, v):
        pass

    @abstractmethod
    def proj(self, x, v):
        pass

    @abstractmethod
    def inner_product(self, u, v):
        pass

    def norm(self, v):
        return np.sqrt(self.inner_product(v, v))

    @abstractmethod
    def transport(self, x1, x2, v1):
        pass


class EuclideanManifold(Manifold):
    def exp(self, x, v):
        return x + v

    def proj(self, x, v):
        return v

    def inner_product(self, u, v):
        return np.trace(u.T @ v)

    def transport(self, x1, x2, v1):
        return v1


class ObliqueManifold(Manifold):
    def exp(self, x, v):
        x = x/np.linalg.norm(x,axis=0)
        norm_v = np.linalg.norm(v,axis=0)
        return x*np.cos(norm_v) + v*np.sinc(norm_v / np.pi)

    def proj(self, x, v):
        x = x/np.linalg.norm(x,axis=0)
        return v - x * np.sum(x*v, axis=0, keepdims=True)

    def inner_product(self, u, v):
        return np.trace(u.T @ v)

    def transport(self, x1, x2, v1):
        return self.proj(x2, v1)


class SphereManifold(Manifold):
    def exp(self, x, v):
        x = x/np.linalg.norm(x)
        norm_v = np.linalg.norm(v)
        return x*np.cos(norm_v) + v*np.sinc(norm_v / np.pi)

    def proj(self, x, v):
        x = x/np.linalg.norm(x)
        return v - x * (x.T@v).item()

    def inner_product(self, u, v):
        return (u.T @ v).item()

    def transport(self, x1, x2, v1):
        return self.proj(x2, v1)


class ProductManifold(Manifold):
    def __init__(self, manifolds:list[type]):
        assert len(manifolds)>0, "ProductManifold must contain at least one manifold."
        self.manifolds = manifolds

    def exp(self, x, v):
        return [self.manifolds[i].exp(x[i], v[i]) for i in range(len(self.manifolds))]

    def proj(self, x, v):
        return [self.manifolds[i].proj(x[i], v[i]) for i in range(len(self.manifolds))]

    def inner_product(self, u, v):
        return sum(self.manifolds[i].inner_product(u[i], v[i]) for i in range(len(self.manifolds)))

    def transport(self, x1, x2, v1):
        return [self.manifolds[i].transport(x1[i], x2[i], v1[i]) for i in range(len(self.manifolds))]