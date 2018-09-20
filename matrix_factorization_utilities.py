import numpy as np
from scipy.optimize import fmin_cg


def normalize_ratings(ratings):
    """
    Given an array of user ratings, subtract the mean of each product's ratings
    :param ratings: 2d array of user ratings
    :return: (normalized ratings array, the calculated means)
    """
    mean_ratings = np.nanmean(ratings, axis=0)
    return ratings - mean_ratings, mean_ratings


def cost(X, *args):
    """
    Cost function for low rank matrix factorization
    :param X: The matrices being factored (P and Q) rolled up as a contiguous array
    :param args: Array containing (num_users, num_products, num_features, ratings, mask, regularization_amount)
    :return: The cost with the current P and Q matrices
    """
    num_users, num_products, num_features, ratings, mask, regularization_amount = args

    # Unroll P and Q
    P = X[0:(num_users * num_features)].reshape(num_users, num_features)
    Q = X[(num_users * num_features):].reshape(num_products, num_features)
    Q = Q.T

    # Calculate current cost
    return (np.sum(np.square(mask * (np.dot(P, Q) - ratings))) / 2) + ((regularization_amount / 2.0) * np.sum(np.square(Q.T))) + ((regularization_amount / 2.0) * np.sum(np.square(P)))


def gradient(X, *args):
    """
    Calculate the cost gradients with the current P and Q.
    :param X: The matrices being factored (P and Q) rolled up as a contiguous array
    :param args: Array containing (num_users, num_products, num_features, ratings, mask, regularization_amount)
    :return: The gradient with the current X
    """
    num_users, num_products, num_features, ratings, mask, regularization_amount = args

    # Unroll P and Q
    P = X[0:(num_users * num_features)].reshape(num_users, num_features)
    Q = X[(num_users * num_features):].reshape(num_products, num_features)
    Q = Q.T

    # Calculate the current gradients for both P and Q
    P_grad = np.dot((mask * (np.dot(P, Q) - ratings)), Q.T) + (regularization_amount * P)
    Q_grad = np.dot((mask * (np.dot(P, Q) - ratings)).T, P) + (regularization_amount * Q.T)

    # Return the gradients as one rolled-up array as expected by fmin_cg
    return np.append(P_grad.ravel(), Q_grad.ravel())


def low_rank_matrix_factorization(ratings, mask=None, num_features=15, regularization_amount=0.01):
    """
    Factor a ratings array into two latent feature arrays (user features and product features)

    :param ratings: Matrix with user ratings to factor
    :param mask: A binary mask of which ratings are present in the ratings array to factor
    :param num_features: Number of latent features to generate for users and products
    :param regularization_amount: How much regularization to apply
    :return: (P, Q) - the factored latent feature arrays
    """
    num_users, num_products = ratings.shape

    # If no mask is provided, consider all 'NaN' elements as missing and create a mask.
    if mask is None:
        mask = np.invert(np.isnan(ratings))

    # Replace NaN values with zero
    ratings = np.nan_to_num(ratings)

    # Create P and Q and fill with random numbers to start
    np.random.seed(0)
    P = np.random.randn(num_users, num_features)
    Q = np.random.randn(num_products, num_features)

    # Roll up P and Q into a contiguous array as fmin_cg expects
    initial = np.append(P.ravel(), Q.ravel())

    # Create an args array as fmin_cg expects
    args = (num_users, num_products, num_features, ratings, mask, regularization_amount)

    # Call fmin_cg to minimize the cost function and this find the best values for P and Q
    X = fmin_cg(cost, initial, fprime=gradient, args=args, maxiter=3000)

    # Unroll the new P and new Q arrays out of the contiguous array returned by fmin_cg
    nP = X[0:(num_users * num_features)].reshape(num_users, num_features)
    nQ = X[(num_users * num_features):].reshape(num_products, num_features)

    return nP, nQ.T


def RMSE(real, predicted):
    """
    Calculate the root mean squared error between a matrix of real ratings and predicted ratings
    :param real: A matrix containing the real ratings (with 'NaN' for any missing elements)
    :param predicted: A matrix of predictions
    :return: The RMSE as a float
    """
    return np.sqrt(np.nanmean(np.square(real - predicted)))