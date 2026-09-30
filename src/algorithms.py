"""Bandit algorithms used in the original GAdaOFUL simulation."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist

from src.config import ExperimentConfig
from src.utils import (
    constraint12,
    func,
    gradient_function,
    init_arms,
    objective_function,
    reward_function,
)


def run_gadaoful(config: ExperimentConfig) -> np.ndarray:
    """Run one GAdaOFUL simulation and return cumulative regret."""
    dim = config.dim
    sigma = config.sigma
    corruption = config.corruption
    T = config.T
    actions = config.actions
    norm = config.norm
    bmu = config.bmu

    cur_crr = 1

    # Preserved from the original notebook because it consumes random draws.
    decision_t = init_arms(dim, norm, actions)
    _ = decision_t

    B = 1
    L = 1
    K = 1 / 4
    k = 1 / 6
    delta = 0.001
    lambda_ = 0.1

    H = np.eye(dim) * lambda_
    theta = np.zeros(dim)
    beta = np.sqrt(lambda_)

    m_0 = 1
    m_1 = 1 / (42 * np.log(2 * T**2 / delta))

    sigma_min = 1 / np.sqrt(T)
    kappa = dim * np.log(
        1 + T / (dim * lambda_ * sigma_min**2)
    )
    tau_0 = 1

    alpha = max(
        np.sqrt(K) / (m_1**0.25 * dim**0.25),
        corruption**0.5 * kappa ** (-0.25),
    )

    sigma_ = [None] * (T + 1)
    w_ = [None] * (T + 1)
    tau_ = [None] * (T + 1)
    y_ = [None] * (T + 1)
    var_ = [None] * (T + 1)
    phi_ = [None] * (T + 1)

    regret = 0.0
    total_regret = []

    for t in range(1, T + 1):
        noise = np.random.randn(actions) * sigma

        flag = 0
        if cur_crr < corruption:
            flag = 1

        decision = init_arms(dim, norm, actions)

        # These random initializations are preserved from the notebook
        # because they affect the subsequent random-number stream.
        rewardS = np.random.randn(actions)
        reward = np.random.randn(actions)

        optimal_reward = float("-inf")

        for arm in range(actions):
            rewardS[arm] = (
                noise[arm] + np.dot(decision[arm], bmu)
            )
            reward[arm] = rewardS[arm]

            if np.dot(decision[arm], bmu) > optimal_reward:
                optimal_reward = np.dot(decision[arm], bmu)

        if flag == 1:
            cur_crr += 1

            for arm in range(actions):
                reward[arm] = (
                    noise[arm] - np.dot(decision[arm], bmu)
                )

        max_dot_product = float("-inf")
        best_i = None

        for i in range(actions):

            def objective(theta_):
                return -np.dot(decision[i], theta_)

            def constraint1(theta_):
                return B - np.linalg.norm(theta_)

            def constraint2(theta_):
                difference = theta_ - theta
                quadratic = np.dot(
                    np.dot(difference, H),
                    difference,
                )
                return beta - np.sqrt(quadratic)

            theta0 = np.zeros_like(theta)

            constraints = [
                {"type": "ineq", "fun": constraint1},
                {"type": "ineq", "fun": constraint2},
            ]

            result = minimize(
                objective,
                theta0,
                constraints=constraints,
            )

            optimal_value = -result.fun

            if optimal_value > max_dot_product:
                max_dot_product = optimal_value
                best_i = i

        phi_[t] = decision[best_i]

        regret += (
            func(optimal_reward)
            - func(np.dot(phi_[t], bmu))
        )

        phi_t_H_inv_norm = np.sqrt(
            np.dot(
                np.dot(phi_[t], np.linalg.inv(H)),
                phi_[t],
            )
        )

        y_[t] = reward_function(
            phi_[t],
            bmu,
            flag,
        )

        var_[t] = sigma

        sigma_[t] = max(
            var_[t],
            sigma_min,
            phi_t_H_inv_norm / m_0,
            alpha * phi_t_H_inv_norm**0.5,
        )

        w_[t] = phi_t_H_inv_norm / sigma_[t]

        tau_[t] = (
            tau_0
            * np.sqrt(1 + w_[t] ** 2)
            / w_[t]
        )

        theta = minimize(
            fun=objective_function,
            x0=theta,
            args=(
                lambda_,
                B,
                k,
                t,
                phi_,
                y_,
                sigma_,
                tau_,
                func,
            ),
            jac=gradient_function,
            constraints=[
                {
                    "type": "ineq",
                    "fun": constraint12,
                }
            ],
        ).x

        H += (
            np.outer(phi_[t], phi_[t])
            / sigma_[t] ** 2
        )

        beta = 1.0
        total_regret.append(regret)

    return np.asarray(total_regret)


def run_cw_oful(config: ExperimentConfig) -> np.ndarray:
    """Run one CW-OFUL simulation and return cumulative regret."""
    dim = config.dim
    sigma = config.sigma
    corruption = config.corruption
    T = config.T
    actions = config.actions
    norm = config.norm
    bmu = config.bmu

    cur_crr = 0

    # Preserved because it consumes random draws in the original notebook.
    decision_t = init_arms(dim, norm, actions)
    _ = decision_t

    LAMBDA = 1
    beta = 1.5
    alpha = 0.2

    sigma_matrix = LAMBDA * np.diag(np.ones(dim))
    BB = np.zeros(dim)

    regret = 0.0
    total_regret = []

    for _ in range(T):
        weight = 1

        noise = (
            t_dist.rvs(df=3, size=actions)
            * sigma
        )

        flag = 0

        if cur_crr < corruption:
            flag = 1

        decision = init_arms(
            dim,
            norm,
            actions,
        )

        rewardS = np.random.randn(actions)
        reward = np.random.randn(actions)

        optimal_reward = float("-inf")

        for arm in range(actions):
            rewardS[arm] = (
                noise[arm]
                + np.dot(decision[arm], bmu)
            )
            reward[arm] = rewardS[arm]

            if np.dot(decision[arm], bmu) > optimal_reward:
                optimal_reward = np.dot(
                    decision[arm],
                    bmu,
                )

        if flag == 1:
            cur_crr += 1

            for arm in range(actions):
                reward[arm] = (
                    noise[arm]
                    - np.dot(decision[arm], bmu)
                )

        hattheta = np.linalg.lstsq(
            sigma_matrix,
            BB,
            rcond=-1,
        )[0]

        hattheta.shape = dim

        max_reward = float("-inf")

        for a_t in range(actions):
            action_t = decision[a_t]

            UU = np.linalg.lstsq(
                sigma_matrix,
                action_t,
                rcond=-1,
            )[0]

            UU.shape = dim

            r_action_t = (
                np.dot(action_t, hattheta)
                + beta
                * np.sqrt(np.dot(UU, action_t))
            )

            if max_reward < r_action_t:
                max_reward = r_action_t
                final_a_t = a_t
                action = action_t
                weight = min(
                    1,
                    alpha
                    / np.sqrt(
                        np.dot(UU, action_t)
                    ),
                )

        regret += (
            func(optimal_reward)
            - func(np.dot(action, bmu))
        )

        sigma_matrix += (
            weight * np.outer(action, action)
        )

        BB += (
            weight
            * reward[final_a_t]
            * action
        )

        total_regret.append(regret)

    return np.asarray(total_regret)


def run_greedy(config: ExperimentConfig) -> np.ndarray:
    """Run one greedy simulation and return cumulative regret."""
    dim = config.dim
    sigma = config.sigma
    corruption = config.corruption
    T = config.T
    actions = config.actions
    norm = config.norm
    bmu = config.bmu

    cur_crr = 0

    decision_t = init_arms(
        dim,
        norm,
        actions,
    )

    LAMBDA = 1
    beta = 10
    alpha = 1
    _ = (beta, alpha)

    sigma_matrix = (
        LAMBDA * np.diag(np.ones(dim))
    )

    BB = np.zeros(dim)

    regret = 0.0
    total_regret = []

    for _ in range(T):
        noise = (
            t_dist.rvs(df=3, size=actions)
            * sigma
        )

        flag = 0

        # This apparently redundant block is preserved
        # from the original notebook because its random
        # draws affect subsequent simulation values.
        if cur_crr < corruption:
            decision = init_arms(
                dim,
                norm,
                actions,
            )

            if cur_crr < corruption:
                flag = 1
        else:
            decision = decision_t

        decision = init_arms(
            dim,
            norm,
            actions,
        )

        rewardS = np.random.randn(actions)
        reward = np.random.randn(actions)

        optimal_reward = float("-inf")

        for arm in range(actions):
            rewardS[arm] = (
                noise[arm]
                + np.dot(decision[arm], bmu)
            )

            reward[arm] = rewardS[arm]

            if np.dot(decision[arm], bmu) > optimal_reward:
                optimal_reward = np.dot(
                    decision[arm],
                    bmu,
                )

        if flag == 1:
            cur_crr += 1

            for arm in range(actions):
                reward[arm] = (
                    noise[arm]
                    - np.dot(decision[arm], bmu)
                )

        hattheta = np.linalg.lstsq(
            sigma_matrix,
            BB,
            rcond=-1,
        )[0]

        hattheta.shape = dim

        max_reward = float("-inf")

        for a_t in range(actions):
            action_t = decision[a_t]

            r_action_t = np.dot(
                action_t,
                hattheta,
            )

            if max_reward < r_action_t:
                max_reward = r_action_t
                final_a_t = a_t
                action = action_t

        regret += (
            func(optimal_reward)
            - func(np.dot(action, bmu))
        )

        sigma_matrix += np.outer(
            action,
            action,
        )

        BB += reward[final_a_t] * action

        total_regret.append(regret)

    return np.asarray(total_regret)


def run_oful(config: ExperimentConfig) -> np.ndarray:
    """Run one OFUL simulation and return cumulative regret."""
    dim = config.dim
    sigma = config.sigma
    corruption = config.corruption
    T = config.T
    actions = config.actions
    norm = config.norm
    bmu = config.bmu

    cur_crr = 0

    decision_t = init_arms(
        dim,
        norm,
        actions,
    )

    LAMBDA = 1
    beta = 1
    alpha = 1
    _ = alpha

    sigma_matrix = (
        LAMBDA * np.diag(np.ones(dim))
    )

    BB = np.zeros(dim)

    regret = 0.0
    total_regret = []

    for _ in range(T):
        noise = (
            t_dist.rvs(df=3, size=actions)
            * sigma
        )

        flag = 0

        # Preserved exactly in spirit from the notebook;
        # the first generated decision is immediately
        # replaced but still changes the RNG stream.
        if cur_crr < corruption:
            decision = init_arms(
                dim,
                norm,
                actions,
            )

            if cur_crr < corruption:
                flag = 1
        else:
            decision = decision_t

        decision = init_arms(
            dim,
            norm,
            actions,
        )

        rewardS = np.random.randn(actions)
        reward = np.random.randn(actions)

        optimal_reward = float("-inf")

        for arm in range(actions):
            rewardS[arm] = (
                noise[arm]
                + np.dot(decision[arm], bmu)
            )

            reward[arm] = rewardS[arm]

            if np.dot(decision[arm], bmu) > optimal_reward:
                optimal_reward = np.dot(
                    decision[arm],
                    bmu,
                )

        if flag == 1:
            cur_crr += 1

            for arm in range(actions):
                reward[arm] = (
                    noise[arm]
                    - np.dot(decision[arm], bmu)
                )

        hattheta = np.linalg.lstsq(
            sigma_matrix,
            BB,
            rcond=-1,
        )[0]

        hattheta.shape = dim

        max_reward = float("-inf")

        for a_t in range(actions):
            action_t = decision[a_t]

            UU = np.linalg.lstsq(
                sigma_matrix,
                action_t,
                rcond=-1,
            )[0]

            UU.shape = dim

            r_action_t = (
                np.dot(action_t, hattheta)
                + beta
                * np.sqrt(np.dot(UU, action_t))
            )

            if max_reward < r_action_t:
                max_reward = r_action_t
                final_a_t = a_t
                action = action_t

        regret += (
            func(optimal_reward)
            - func(np.dot(action, bmu))
        )

        sigma_matrix += np.outer(
            action,
            action,
        )

        BB += reward[final_a_t] * action

        total_regret.append(regret)

    return np.asarray(total_regret)


def run_adaoful(config: ExperimentConfig) -> np.ndarray:
    """Run one AdaOFUL simulation and return cumulative regret."""
    dim = config.dim
    sigma = config.sigma
    corruption = config.corruption
    T = config.T
    actions = config.actions
    norm = config.norm
    bmu = config.bmu

    cur_crr = 1

    # Preserved because the original notebook makes this draw.
    decision_t = init_arms(
        dim,
        norm,
        actions,
    )
    _ = decision_t

    # In the original notebook this value came implicitly
    # from the earlier GAdaOFUL cell.
    B = 1

    L = 1
    K = 1
    k = 1
    delta = 0.001
    lambda_ = 0.1

    H = np.eye(dim) * lambda_
    theta = np.zeros(dim)
    beta = np.sqrt(lambda_)

    m_0 = 1
    m_1 = 1 / (
        42 * np.log(2 * T**2 / delta)
    )

    sigma_min = 1 / np.sqrt(T)

    kappa = dim * np.log(
        1
        + T
        / (
            dim
            * lambda_
            * sigma_min**2
        )
    )

    tau_0 = 1

    alpha = max(
        np.sqrt(K)
        / (
            m_1**0.25
            * dim**0.25
        ),
        0,
    )

    sigma_ = [None] * (T + 1)
    w_ = [None] * (T + 1)
    tau_ = [None] * (T + 1)
    y_ = [None] * (T + 1)
    var_ = [None] * (T + 1)
    phi_ = [None] * (T + 1)

    regret = 0.0
    total_regret = []

    for t in range(1, T + 1):
        noise = (
            np.random.randn(actions)
            * sigma
        )

        flag = 0

        if cur_crr < corruption:
            flag = 1

        decision = init_arms(
            dim,
            norm,
            actions,
        )

        rewardS = np.random.randn(actions)
        reward = np.random.randn(actions)

        optimal_reward = float("-inf")

        for arm in range(actions):
            rewardS[arm] = (
                noise[arm]
                + np.dot(decision[arm], bmu)
            )

            reward[arm] = rewardS[arm]

            if np.dot(decision[arm], bmu) > optimal_reward:
                optimal_reward = np.dot(
                    decision[arm],
                    bmu,
                )

        if flag == 1:
            cur_crr += 1

            for arm in range(actions):
                reward[arm] = (
                    noise[arm]
                    - np.dot(decision[arm], bmu)
                )

        max_dot_product = float("-inf")
        best_i = None

        for i in range(actions):

            def objective(theta_):
                return -np.dot(
                    decision[i],
                    theta_,
                )

            def constraint1(theta_):
                return (
                    B
                    - np.linalg.norm(theta_)
                )

            def constraint2(theta_):
                difference = theta_ - theta

                quadratic = np.dot(
                    np.dot(
                        difference,
                        H,
                    ),
                    difference,
                )

                return (
                    beta
                    - np.sqrt(quadratic)
                )

            theta0 = np.zeros_like(theta)

            constraints = [
                {
                    "type": "ineq",
                    "fun": constraint1,
                },
                {
                    "type": "ineq",
                    "fun": constraint2,
                },
            ]

            result = minimize(
                objective,
                theta0,
                constraints=constraints,
            )

            optimal_value = -result.fun

            if optimal_value > max_dot_product:
                max_dot_product = optimal_value
                best_i = i

        phi_[t] = decision[best_i]

        regret += (
            func(optimal_reward)
            - func(np.dot(phi_[t], bmu))
        )

        phi_t_H_inv_norm = np.sqrt(
            np.dot(
                np.dot(
                    phi_[t],
                    np.linalg.inv(H),
                ),
                phi_[t],
            )
        )

        y_[t] = reward_function(
            phi_[t],
            bmu,
            flag,
        )

        var_[t] = sigma

        sigma_[t] = max(
            var_[t],
            sigma_min,
            phi_t_H_inv_norm / m_0,
            alpha
            * phi_t_H_inv_norm**0.5,
        )

        w_[t] = (
            phi_t_H_inv_norm
            / sigma_[t]
        )

        tau_[t] = (
            tau_0
            * np.sqrt(1 + w_[t] ** 2)
            / w_[t]
        )

        theta = minimize(
            fun=objective_function,
            x0=theta,
            args=(
                lambda_,
                B,
                k,
                t,
                phi_,
                y_,
                sigma_,
                tau_,
                func,
            ),
            jac=gradient_function,
            constraints=[
                {
                    "type": "ineq",
                    "fun": constraint12,
                }
            ],
        ).x

        H += (
            np.outer(
                phi_[t],
                phi_[t],
            )
            / sigma_[t] ** 2
        )

        beta = 1.0

        total_regret.append(regret)

    return np.asarray(total_regret)
