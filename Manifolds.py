import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    @abstractmethod
    def exp(self, x, v):
        pass

    @abstractmethod
    def proj(self, x, v):
        pass


class EuclideanManifold(Manifold):
    def exp(self, x, v):
        return x + v

    def proj(self, x, v):
        return v


class UnitQuaternionManifold(Manifold):
    def exp(self, x, v):
        norm_v = np.linalg.norm(v)
        return x*np.cos(norm_v) + v/norm_v*np.sin(norm_v)

    def proj(self, x, v):
        return (np.eye(4) - x@x.T) @ v


class ProductManifold(Manifold):
    def __init__(self, manifolds:list[type]):
        assert len(manifolds)>0, "ProductManifold must contain >1 manifold."
        self.manifolds = manifolds

    def exp(self, x, v):
        return [self.manifolds[i].exp(x[i], v[i]) for i in range(len(self.manifolds))]

    def proj(self, x, v):
        return [self.manifolds[i].proj(x[i], v[i]) for i in range(len(self.manifolds))]