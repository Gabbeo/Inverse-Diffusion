import numpy as np


class Gauss:
    @staticmethod
    def weighted_filters_1d(sigmas, direction="x"):
        K = len(sigmas)
        assert K % 2 == 1
        filter_matrices = []

        for i in range(K - 1):
            fourth_root_delta_k = np.sqrt(np.sqrt(sigmas[i + 1] - sigmas[i]))
            sigma_line = (sigmas[i + 1] + sigmas[i]) / 2
            filter_matrix = fourth_root_delta_k * Gauss.filter_1d(sigma_line, direction)
            filter_matrices.append(filter_matrix)

        return filter_matrices

    @staticmethod
    def filter_1d(sigma, direction="x"):
        size = 6 * int(np.ceil(sigma)) + 1
        center_point = np.floor(size / 2)

        if direction == "x":
            gauss_filter = np.zeros([1, size, 1, 1])
            for i in range(size):
                dist_x = np.abs(center_point - i)
                gauss_filter[0][i] = Gauss.__1D_gaussian_kernel(sigma, dist_x)
        elif direction == "y":
            gauss_filter = np.zeros([size, 1, 1, 1])
            for i in range(size):
                dist_y = np.abs(center_point - i)
                gauss_filter[i][0] = Gauss.__1D_gaussian_kernel(sigma, dist_y)

        gauss_filter = np.asarray(gauss_filter).astype(np.float32)
        gauss_filter *= 1 / np.sum(gauss_filter)
        return gauss_filter

    @staticmethod
    def __1D_gaussian_kernel(sigma, x):
        division = 1 / (np.sqrt(2 * np.pi) * sigma)
        exponential = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return division * exponential
