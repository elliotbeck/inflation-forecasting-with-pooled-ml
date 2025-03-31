import numpy as np


def get_ewma_moments(Y, alpha=0.15, eps=0.01, bw=12):
    n, p = Y.shape

    # Compute EWMA weights
    while True:
        w = alpha * (1 - alpha) ** np.arange(n)[::-1]
        total_weight = np.sum(w)
        if 1 - total_weight < eps:
            break
        alpha *= 1.05  # increase alpha gradually (5% each time)
        if alpha >= 1:
            raise ValueError(
                "Alpha reached 1; can't ensure weights decay fast enough for this sample size."
            )
    w /= np.sum(w)

    # Weight matrix for broadcasting
    w_mat = np.tile(w.reshape(-1, 1), (1, p))

    # EWMA mean vector
    mu_ewma = np.sum(w_mat * Y, axis=0)

    # Shrinkage target for mean: global mean
    target = np.ones(p) * np.mean(mu_ewma)

    # Gamma hat for mean shrinkage
    gamma_hat = np.sum((mu_ewma - target) ** 2)

    # Centered data using equal weights
    mu_ew = np.mean(Y, axis=0)
    Y_c_ew = Y - mu_ew

    # Compute w_pi weights
    beta = 1 - alpha
    w_pi = alpha**2 / (1 - beta**2) * beta ** np.arange(bw + 1)
    if bw > 0:
        w_pi[1:] *= 2

    # Estimate pi_hat for mean shrinkage
    pi_hat = w_pi[0] * np.sum(Y_c_ew**2) / n
    for k in range(1, bw + 1):
        Y_cross = Y_c_ew[:-k] * Y_c_ew[k:]
        pi_hat += w_pi[k] * np.sum(Y_cross) / n

    # Shrinkage intensity for mean
    kappa_hat = pi_hat / (pi_hat + gamma_hat)
    shrinkage = np.clip(kappa_hat, 0, 1)

    # Shrinkage mean estimator
    mu_hat = shrinkage * target + (1 - shrinkage) * mu_ewma

    # Recenter using equal weights
    mu_ew = np.mean(Y, axis=0)
    Y_c_ew = Y - mu_ew

    # EWMA covariance matrix
    sigma_ewma = Y_c_ew.T @ (Y_c_ew * w_mat)

    # Shrinkage targets for covariance
    identity = np.eye(p)
    one_p = np.ones((p, p))
    mean_var = np.mean(np.diag(sigma_ewma))
    mean_covar = np.sum(sigma_ewma - np.diag(np.diag(sigma_ewma))) / (p * (p - 1))
    target1 = mean_var * identity
    target2 = mean_var * identity + mean_covar * (one_p - identity)

    # Gamma hats
    gamma_hat1 = np.linalg.norm(sigma_ewma - target1) ** 2
    gamma_hat2 = np.linalg.norm(sigma_ewma - target2) ** 2

    # Sample covariance matrix
    sigma_ew = (Y_c_ew.T @ Y_c_ew) / n
    sigma_ew_squared = sigma_ew**2

    # Pi hat for covariance shrinkage
    help_cross = Y_c_ew**2
    mat_prod_cross = (help_cross.T @ help_cross) / n
    mat_diff = mat_prod_cross - sigma_ew_squared
    pi_hat = w_pi[0] * np.sum(mat_diff)

    for k in range(1, bw + 1):
        help_cross = Y_c_ew[:-k] * Y_c_ew[k:]
        mat_prod_cross = (help_cross.T @ help_cross) / n
        mat_diff = mat_prod_cross - ((n - k) / n) * sigma_ew_squared
        pi_hat += w_pi[k] * np.sum(mat_diff)

    # Shrinkage intensities for covariance
    kappa_hat1 = pi_hat / (pi_hat + gamma_hat1)
    shrinkage1 = np.clip(kappa_hat1, 0, 1)
    kappa_hat2 = pi_hat / (pi_hat + gamma_hat2)
    shrinkage2 = np.clip(kappa_hat2, 0, 1)

    # Shrinkage covariance estimators
    sigma_hat1 = shrinkage1 * target1 + (1 - shrinkage1) * sigma_ewma
    sigma_hat2 = shrinkage2 * target2 + (1 - shrinkage2) * sigma_ewma

    return {
        "mu_ewma": mu_ewma,
        "sigma_ewma": sigma_ewma,
        "mu_hat": mu_hat,
        "shrinkage_mu": shrinkage,
        "sigma_hat1": sigma_hat1,
        "sigma_hat2": sigma_hat2,
        "shrinkage_sigma1": shrinkage1,
        "shrinkage_sigma2": shrinkage2,
    }
