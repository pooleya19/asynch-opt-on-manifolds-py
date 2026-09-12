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


class ObliqueManifold(Manifold):
    def exp(self, x, v):
        x = x/np.linalg.norm(x,axis=0)
        norm_v = np.linalg.norm(v,axis=0)
        return x*np.cos(norm_v) + v*np.sinc(norm_v / np.pi)

    def proj(self, x, v):
        x = x/np.linalg.norm(x,axis=0)
        return v - x * np.sum(x*v, axis=0, keepdims=True)


class SphereManifold(Manifold):
    def exp(self, x, v):
        x = x/np.linalg.norm(x)
        norm_v = np.linalg.norm(v)
        return x*np.cos(norm_v) + v*np.sinc(norm_v / np.pi)

    def proj(self, x, v):
        x = x/np.linalg.norm(x)
        return v - x * (x.T@v).item()


class UnitQuaternionManifold(Manifold):
    def exp(self, x, v):
        x = x/np.linalg.norm(x)
        norm_v = np.linalg.norm(v)
        return x*np.cos(norm_v) + v*np.sin(norm_v) / (norm_v if norm_v > 0 else 1)

    def proj(self, x, v):
        x = x/np.linalg.norm(x)
        return (np.eye(4) - x@x.T) @ v


class ProductManifold(Manifold):
    def __init__(self, manifolds:list[type]):
        assert len(manifolds)>0, "ProductManifold must contain >1 manifold."
        self.manifolds = manifolds

    def exp(self, x, v):
        return [self.manifolds[i].exp(x[i], v[i]) for i in range(len(self.manifolds))]

    def proj(self, x, v):
        return [self.manifolds[i].proj(x[i], v[i]) for i in range(len(self.manifolds))]