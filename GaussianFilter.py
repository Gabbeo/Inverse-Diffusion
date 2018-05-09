import numpy as np


class Gauss:
    """
    Creates a list containing the corresponding filters for the array of sigmas provided to the function.
    The filters will be in the direction specified.
    """
    @staticmethod
    def weighted_filters_1d(sigmas, direction="x"):
        K = len(sigmas)
        assert K % 2 == 1  # Since we use the delta difference between sigmas we need an odd amount.
        filter_matrices = []

        # Loop over all sigma-deltas and append a new filter for each delta.
        for i in range(K - 1):
            # We take the fourth root because we have separated the filters.
            fourth_root_delta_k = np.sqrt(np.sqrt(sigmas[i + 1] - sigmas[i]))
            sigma_bar = (sigmas[i + 1] + sigmas[i]) / 2
            filter_matrix = fourth_root_delta_k * Gauss.filter_1d(sigma_bar, direction)
            filter_matrices.append(filter_matrix)

        return filter_matrices

    """
    Creates a one dimensional Gaussian filter using the Gaussian kernel.
    The function returns a NumPy array in the specified direction (eg. 1 x (6 * sigma + 1)).
    """
    @staticmethod
    def filter_1d(sigma, direction="x"):
        # Rule of thumb for filter sizes.
        size = 6 * int(np.ceil(sigma)) + 1
        # Calculates the center point for use when calculating the value of the different locations in the array.
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
        # Normalize the filter.
        gauss_filter *= 1 / np.sum(gauss_filter)
        return gauss_filter

    """
    This function is simply the Gaussian kernel in one dimension.
    See the Wikipedia page for the definition.
    """
    @staticmethod
    def __1D_gaussian_kernel(sigma, x):
        division = 1 / (np.sqrt(2 * np.pi) * sigma)
        exponential = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return division * exponential
