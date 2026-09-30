"""Shared utility functions for the GAdaOFUL simulations."""

import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import t as t_dist


def init_vector(dim, norm):
    """Generate a random vector with coordinates scaled by norm / sqrt(dim)."""
    vec = 2 * np.random.rand(dim) - 1.0
    return vec * norm / math.sqrt(dim)


def init_arms(dim, norm, num):
    """Generate a collection of random action vectors."""
    decision = np.random.rand(num, dim)
    for i in range(num):
        decision[i] = init_vector(dim, norm)
    return decision


def func(x):
    """Nonlinear reward link function."""
    return np.exp(x)


def reward_function(chosen_arm, theta_star, flag, df=3, scale=1):
    """Generate an observed heavy-tailed reward."""
    expected_payoff = func(chosen_arm.dot(theta_star))
    noise = t_dist.rvs(df=df) * scale

    if flag == 0:
        observed_payoffs = expected_payoff + noise
    else:
        observed_payoffs = -expected_payoff + noise

    return observed_payoffs


def z_s(u, y_s, sigma_s, f):
    """Compute the standardized residual used in the robust loss."""
    return (y_s - f(u)) / sigma_s


def objective_function(
    theta,
    lambda_k,
    B,
    k,
    t,
    phi_s,
    y_s,
    sigma_s,
    tau_s,
    f,
):
    """Compute the robust objective used by GAdaOFUL and AdaOFUL."""
    loss = lambda_k * k / 2 * np.linalg.norm(theta) ** 2

    for s in range(1, t + 1):
        integral_func = (
            lambda u: (
                tau_s[s] * z_s(u, y_s[s], sigma_s[s], f)
            )
            / np.sqrt(
                tau_s[s] ** 2
                + z_s(u, y_s[s], sigma_s[s], f) ** 2
            )
        )

        integral_result, _ = quad(
            integral_func,
            0,
            np.dot(phi_s[s], theta),
        )

        loss += -1 / sigma_s[s] * integral_result

    return loss


def gradient_function(
    theta,
    lambda_k,
    B,
    k,
    t,
    phi_s,
    y_s,
    sigma_s,
    tau_s,
    f,
):
    """Compute the gradient of the robust objective."""
    grad = lambda_k * k * theta

    for s in range(1, t + 1):
        z_at_upper_limit = z_s(
            np.dot(phi_s[s], theta),
            y_s[s],
            sigma_s[s],
            f,
        )

        grad_term = (
            tau_s[s] * z_at_upper_limit
        ) / np.sqrt(
            tau_s[s] ** 2 + z_at_upper_limit ** 2
        )

        grad_integral_part = (
            -1 / sigma_s[s] * phi_s[s] * grad_term
        )

        grad = grad + grad_integral_part

    return grad


def constraint12(theta):
    """Return the unit-ball inequality constraint value."""
    return 1 - np.linalg.norm(theta)


def project_to_unit_ball(theta):
    """Project a vector onto the Euclidean unit ball."""
    vector_norm = np.linalg.norm(theta)

    if vector_norm > 1:
        return theta / vector_norm

    return theta
