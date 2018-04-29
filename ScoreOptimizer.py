import numpy as np


# This is an algorithm for calculating the optimal F1-score given true coordinates and
# calculated coordinates with their corresponding pseudo-likelihoods.
# All credit goes to Pol del Aguila Pla for devising this algorithm.
def optimal_f1_score(true_coordinates, calculated_coordinates, pseudo_likelihood, gate):
    detected_number = len(calculated_coordinates)
    true_number = len(true_coordinates)

    # Sorts pseudo-likelihoods in descending order and sorts corresponding calculated coordinates in the same order.
    pseudo_likelihood_indices = np.argsort(pseudo_likelihood)[::-1]
    pseudo_likelihood = np.sort(pseudo_likelihood)[::-1]
    calculated_coordinates = [calculated_coordinates[i] for i in pseudo_likelihood_indices]

    # Defines variables for correct detections and marking true coordinates that have already been used.
    correct = np.zeros([detected_number, 1], dtype=bool)
    taken_by = (-1) * np.ones([true_number, 1])  # Creates array with all elements as NaN.

    for i in range(detected_number):
        # Calculates square distance from a calculated detection to all true coordinates.
        distances = np.square(true_coordinates[:, 0] - calculated_coordinates[i][0] * np.ones([1, true_number])) \
            + np.square(true_coordinates[:, 1] - calculated_coordinates[i][1] * np.ones([1, true_number]))

        # Sets the true coordinates' corresponding distances to infinity.
        for ii in range(len(distances[0])):
            if taken_by[ii] != (-1):
                distances[0][ii] = np.infty

        min_index = np.argmin(distances)
        min_dist = distances[0][min_index]

        # Calculated coordinates can only be matched to true coordinates if distance between them is below gate value.
        if np.sqrt(min_dist) <= gate:
            correct[i] = True
            taken_by[min_index] = True

    # Calculations for precision, recall and F1-scores are adjusted to avoid division by zero.
    TP = np.cumsum(correct)  # True positives.
    FP = np.arange(1, detected_number + 1) - np.cumsum(correct)  # False positives.
    FN = true_number - np.cumsum(correct)  # False negatives.
    precision = np.divide(TP, (TP + FP + np.finfo(float).eps))
    recall = np.divide(TP, (TP + FN + np.finfo(float).eps))
    f1_scores = 2 * np.multiply(precision,  np.divide(recall, (precision + recall + np.finfo(float).eps)))

    optimal_index = np.argmax(f1_scores)
    f1_score = f1_scores[optimal_index]

    if optimal_index == detected_number:
        optimal_threshold = -np.infty
    else:
        optimal_threshold = .5 * (pseudo_likelihood[optimal_index] + pseudo_likelihood[optimal_index + 1])

    return optimal_threshold, f1_score
