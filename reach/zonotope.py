import numpy as np
from itertools import product


class Zonotope:
    """
    A class representing a zonotope.

    Attributes:
        center (np.ndarray): Center of the zonotope, shape (n,).
        generators (np.ndarray): Generators of the zonotope, shape (n, m),
            where n is the dimension and m is the number of generators.
    """
    class Random:

        def __init__(self, outer):
            self.outer = outer

        @staticmethod
        def farthest_point_sampling(nsamples, num_dims, bounds=None,
                                    n_candidates=10):
            if bounds is None:
                bounds = [(0, 1)] * num_dims

            rng = np.random.default_rng(1)
            lower_bounds, upper_bounds = np.array(bounds).T

            # Preallocate output
            points = np.empty((nsamples, num_dims), dtype=np.float32)

            # Initialize with n_candidates random points
            init = rng.uniform(lower_bounds, upper_bounds,
                               (n_candidates, num_dims))
            points[:n_candidates] = init
            n_filled = n_candidates

            # Keep track of min distances from all future candidates to
            #  existing points
            while n_filled < nsamples:
                batch_size = min(100, nsamples - n_filled + n_candidates)
                candidates = rng.uniform(
                    lower_bounds, upper_bounds, (batch_size, num_dims))

                # Compute pairwise distances between current points and
                #  candidates
                dists = np.linalg.norm(
                    points[:n_filled, None, :] - candidates[None, :, :],
                    axis=2)
                min_dists = np.min(dists, axis=0)

                # Pick top k farthest candidates
                k = min(n_candidates, nsamples - n_filled)
                idx = np.argpartition(min_dists, -k)[-k:]
                points[n_filled:n_filled + k] = candidates[idx]
                n_filled += k

            return points

        def uniform(self, nsamples):
            n_generators = self.outer.generators.shape[1]
            alpha = np.random.rand(nsamples, n_generators)

            alpha = 2*(alpha-0.5)
            points = self.outer.center + alpha@self.outer.generators.T
            return points

        def farthest(self, nsamples, n_candidates=10):
            n_generators = self.outer.generators.shape[1]
            alpha = self.farthest_point_sampling(
                nsamples, n_generators, n_candidates=n_candidates)  # [0, 1]

            alpha = 2*(alpha-0.5)  # [-1, 1]
            points = self.outer.center + alpha@self.outer.generators.T
            return points

        def boundary(self, nsamples):
            n_generators = self.outer.generators.shape[1]
            alpha = np.random.randint(0, 2, size=(nsamples, n_generators))
            for i in range(nsamples):
                whichalpha = np.random.randint(0, n_generators)
                alpha[i, whichalpha] = np.random.rand()

            alpha = 2*(alpha-0.5)  # [-1, 1]
            points = self.outer.center + alpha@self.outer.generators.T
            return points

    def __init__(self, center, generators):
        self.center = center
        self.generators = generators
        self.rand = self.Random(self)

    def get_vertices(self):

        alphas = list(product((-1, 1), repeat=self.generators.shape[1]))
        points = np.array(
            [np.sum(self.generators*alpha, axis=1) for alpha in alphas]
        ) + self.center[None, :]  # shape = (npoints, xdim)

        return points

    def project(self, dims):  # project zonotope to a subset of dimensions.
        return Zonotope(self.center[dims], self.generators[dims])


if __name__ == "__main__":

    zonotope = Zonotope(
        np.array([0, 0, 0]),
        np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ]).T
    )

    farthest = zonotope.rand.farthest(1000, n_candidates=50)
    uniform = zonotope.rand.uniform(1000)
    boundary = zonotope.rand.boundary(1000)
    vertices = zonotope.get_vertices()
    projected = zonotope.project([0, 1, 2])

    print()
