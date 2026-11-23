"""Reproduce the finite-grid checks and tables in the SLDI revision."""

import argparse
import csv
from pathlib import Path

import numpy as np


PARAMETERS = np.array([0.8, 0.55, 0.9])
OBSERVATION_SD = 0.30
SUBSTEPS = 4
SEEDS = range(101, 109)
PILOT_SIZES = (128, 1024)
TEST_SIZE = 8192
RIDGE = 0.01


def schedule(observation_times, observations, substeps=SUBSTEPS):
    """Split intervals at every emission. The guide sees the next observation."""
    steps, guide, emit = [], [], []
    for left, right, value in zip(
        np.r_[0.0, observation_times[:-1]], observation_times, observations
    ):
        steps.extend([(right - left) / substeps] * substeps)
        guide.extend([value] * substeps)
        emit.extend([False] * (substeps - 1) + [True])
    return np.array(steps), np.array(guide), np.array(emit)


def nonlinear_paths(noise, times, observations, parameters=PARAMETERS, substeps=SUBSTEPS):
    """Return path ELBOs, exact gradients, and mixed innovation derivatives."""
    steps, guide, emit = schedule(times, observations, substeps=substeps)
    batch_size, nsteps = noise.shape
    z = np.full(batch_size, -0.2)
    tangent = np.zeros((batch_size, len(parameters)))
    gradient = np.zeros_like(tangent)
    elbo = np.zeros(batch_size)
    states, reverse_tape = [], []
    a, b, c = parameters
    tau_sq = OBSERVATION_SD**2

    for k, (dt, y, observe) in enumerate(zip(steps, guide, emit)):
        states.append(np.column_stack((z, tangent)))
        tanh_z = np.tanh(z)
        d_tanh_z = 1 - tanh_z**2
        dd_tanh_z = -2 * tanh_z * d_tanh_z
        sigma = 0.55 + 0.19 * tanh_z
        dsigma = 0.19 * d_tanh_z
        ddsigma = 0.19 * dd_tanh_z
        f = np.tanh(y - z)
        df = -(1 - f**2)
        ddf = -2 * f * (1 - f**2)
        u, du, ddu = c * f, c * df, c * ddf
        root_dt = np.sqrt(dt)
        increment = root_dt * noise[:, k]

        jac = 1 + dt * (-a + b * d_tanh_z + dsigma * u + sigma * du) + dsigma * increment
        hess = dt * (b * dd_tanh_z + ddsigma * u + 2 * dsigma * du + sigma * ddu)
        hess += ddsigma * increment
        direct = dt * np.column_stack((-z, tanh_z, sigma * f))
        mixed = dt * np.column_stack((-np.ones_like(z), d_tanh_z, dsigma * f + sigma * df))

        cost_z = -dt * u * du
        cost_zz = -dt * (du**2 + u * ddu)
        cost_mixed = np.zeros_like(tangent)
        cost_mixed[:, 2] = -2 * dt * c * f * df
        gradient += cost_z[:, None] * tangent
        gradient[:, 2] -= dt * u * f
        elbo -= 0.5 * dt * u**2

        next_z = z + dt * (-a * z + b * tanh_z + sigma * u) + sigma * increment
        next_tangent = jac[:, None] * tangent + direct
        emission_z = (y - next_z) / tau_sq if observe else np.zeros_like(z)
        emission_zz = -1 / tau_sq if observe else 0.0
        if observe:
            elbo -= 0.5 * ((y - next_z)**2 / tau_sq + np.log(2 * np.pi * tau_sq))
            gradient += emission_z[:, None] * next_tangent
        reverse_tape.append((
            sigma, dsigma, tangent, next_tangent, jac, hess, mixed,
            cost_z, cost_zz, cost_mixed, emission_z, emission_zz, root_dt,
        ))
        z, tangent = next_z, next_tangent

    adjoint_z = np.zeros_like(tangent)
    adjoint_tangent = np.zeros_like(tangent)
    derivatives = np.empty((batch_size, nsteps, len(parameters)))
    for k in range(nsteps - 1, -1, -1):
        (sigma, dsigma, tangent, next_tangent, jac, hess, mixed,
         cost_z, cost_zz, cost_mixed, emission_z, emission_zz, root_dt) = reverse_tape[k]
        adjoint_z += emission_zz * next_tangent
        adjoint_tangent += emission_z[:, None]
        derivatives[:, k] = root_dt * (
            sigma[:, None] * adjoint_z + dsigma[:, None] * tangent * adjoint_tangent
        )
        adjoint_z = (
            jac[:, None] * adjoint_z + (hess[:, None] * tangent + mixed) * adjoint_tangent
            + cost_zz[:, None] * tangent + cost_mixed
        )
        adjoint_tangent = jac[:, None] * adjoint_tangent + cost_z[:, None]
    return elbo, gradient, np.stack(states, axis=1), derivatives


def check_derivatives(times, observations):
    noise = np.random.default_rng(71).standard_normal((5, len(times) * SUBSTEPS))
    _, gradient, _, mixed = nonlinear_paths(noise, times, observations)
    step = 1e-5
    for j, direction in enumerate(np.eye(len(PARAMETERS))):
        upper = nonlinear_paths(noise, times, observations, PARAMETERS + step * direction)[0]
        lower = nonlinear_paths(noise, times, observations, PARAMETERS - step * direction)[0]
        np.testing.assert_allclose((upper - lower) / (2 * step), gradient[:, j], atol=2e-7)
    for k in range(noise.shape[1]):
        direction = np.zeros_like(noise)
        direction[:, k] = step
        upper = nonlinear_paths(noise + direction, times, observations)[1]
        lower = nonlinear_paths(noise - direction, times, observations)[1]
        np.testing.assert_allclose((upper - lower) / (2 * step), mixed[:, k], atol=2e-7)


def polynomial_features(state, use_tangent):
    state = state if use_tangent else state[:, :1]
    first, second = np.triu_indices(state.shape[1])
    return np.column_stack((state, state[:, first] * state[:, second]))


def correction(train_state, test_state, targets, test_basis, use_tangent, stepwise=False):
    """Fit independent ridge regressions at each time on pilot paths only."""
    nsamples, nsteps, nparameters = targets[0].shape
    estimate = np.empty((test_state.shape[0], nsteps, nparameters))
    for k in range(nsteps):
        train = polynomial_features(train_state[:, k], use_tangent)
        test = polynomial_features(test_state[:, k], use_tangent)
        spread = train.std(axis=0)
        spread[spread < 1e-10] = 1
        mean = train.mean(axis=0)
        train = np.column_stack((np.ones(nsamples), (train - mean) / spread))
        test = np.column_stack((np.ones(len(test)), (test - mean) / spread))
        penalty = np.eye(train.shape[1]) * RIDGE * nsamples
        penalty[0, 0] = 0
        responses = np.concatenate([target[:, k] for target in targets], axis=1)
        weights = np.linalg.solve(train.T @ train + penalty, train.T @ responses)
        fitted = (test @ weights).reshape(len(test), len(targets), nparameters)
        estimate[:, k] = (test_basis[:, k, :, None] * fitted).sum(axis=1)
    return estimate if stepwise else estimate.sum(axis=1)


def basis(noise):
    return np.stack((noise, (noise**2 - 1) / np.sqrt(2)), axis=-1)


def variance(gradient):
    return gradient.var(axis=0, ddof=1).sum()


def cases():
    """Four fixed contexts from independent synthetic prior paths."""
    rng = np.random.default_rng(8604)
    for case in range(4):
        times = np.r_[np.sort(rng.uniform(0.20, 1.80, 4)), 2.0]
        steps, _, emit = schedule(times, np.zeros(len(times)), substeps=16)
        z = -0.2
        observations = []
        for dt, observe in zip(steps, emit):
            z += (-0.8 * z + 0.55 * np.tanh(z)) * dt
            z += (0.55 + 0.19 * np.tanh(z)) * np.sqrt(dt) * rng.normal()
            if observe:
                observations.append(z + OBSERVATION_SD * rng.normal())
        yield case, times, np.array(observations)


def nonlinear_benchmark():
    rows = []
    for case, times, y in cases():
        check_derivatives(times, y)
        nsteps = len(times) * SUBSTEPS
        for seed in SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([case, seed]))
            pilot_noise = rng.standard_normal((max(PILOT_SIZES), nsteps))
            noise = rng.standard_normal((TEST_SIZE, nsteps))
            _, pilot_gradient, pilot_state, mixed = nonlinear_paths(pilot_noise, times, y)
            _, gradient, test_state, _ = nonlinear_paths(noise, times, y)
            _, negative, _, _ = nonlinear_paths(-noise, times, y)
            original_variance = variance(gradient)
            pilot_basis, test_basis = basis(pilot_noise), basis(noise)
            for size in PILOT_SIZES:
                row = dict(case=case, seed=seed, pilot=size, base_trace=original_variance)
                row["antithetic"] = 2 * variance((gradient + negative) / 2) / original_variance
                first = mixed[:size]
                second = first * pilot_noise[:size, :, None] / np.sqrt(2)
                direct = [pilot_gradient[:size, None, :] * pilot_basis[:size, :, h, None]
                          for h in range(2)]
                for method, targets, tangent in (
                    ("moment", direct, True),
                    ("stein1", [first], True),
                    ("stein2", [first, second], True),
                    ("state_only", [first, second], False),
                ):
                    adjustment = correction(
                        pilot_state[:size], test_state, targets,
                        test_basis[:, :, :len(targets)], tangent,
                    )
                    row[method] = variance(gradient - adjustment) / original_variance
                    row[method + "_shift"] = np.linalg.norm(adjustment.mean(axis=0))
                    row[method + "_shift_se"] = np.sqrt(variance(adjustment) / TEST_SIZE)
                rows.append(row)
    return rows


def ou_benchmark():
    rng = np.random.default_rng(470)
    steps = np.array([0.13, 0.29, 0.11, 0.37, 0.41, 0.24, 0.45])
    multiplier = 1 - 0.65 * steps
    coefficients = np.sqrt(steps) * np.r_[np.cumprod(multiplier[:0:-1])[::-1], 1.0]
    rho = multiplier.prod()
    sigma, tau, x, z0, prior_variance = 0.75, 0.35, 0.70, -0.20, 0.6
    noise = rng.standard_normal((200_000, len(steps)))
    prefix = np.column_stack((np.zeros(len(noise)), np.cumsum(noise * coefficients, axis=1)[:, :-1]))
    w = noise @ coefficients
    raw = ((x - rho * z0) * w - sigma * w**2) / tau**2
    first = (x - rho * z0 - 2 * sigma * prefix) * coefficients / tau**2
    second = -np.sqrt(2) * sigma * coefficients**2 / tau**2
    debiased1 = raw - (first * noise).sum(axis=1)
    debiased2 = debiased1 - ((noise**2 - 1) * second / np.sqrt(2)).sum(axis=1)
    b2 = coefficients @ coefficients
    mean = -sigma * b2 / tau**2
    np.testing.assert_allclose(debiased2, mean, atol=1e-12)
    expected_var = ((x - rho * z0)**2 * b2 + 2 * sigma**2 * b2**2) / tau**4
    expected_residual = 2 * sigma**2 * (coefficients**4).sum() / tau**4
    path_var = sigma**2 * b2
    var_in = rho**2 * prior_variance + tau**2

    def log_density(var):
        return -0.5 * (np.log(2 * np.pi * var) + (x - rho * z0)**2 / var)

    gap = log_density(var_in + path_var) - log_density(var_in) + path_var / (2 * tau**2)
    # Compute the smallest KL achievable by Euler mean shifts with fixed transition variance.
    conditional_loading = coefficients / np.sqrt(steps)
    tail_variance = tau**2 + sigma**2 * (b2 - np.cumsum(coefficients**2))
    tilt = sigma**2 * steps * conditional_loading**2 / tail_variance
    drift_only_gap = 0.5 * (tilt - np.log1p(tilt)).sum()

    # Independently verify it using the joint Gaussian smoother and the
    # covariance of the recognition chain with optimal transition means.
    prior_factor = np.eye(len(steps) + 1)
    prior_factor[0, 0] = np.sqrt(prior_variance)
    for index, dt in enumerate(steps):
        prior_factor[index + 1, :index + 1] = multiplier[index] * prior_factor[index, :index + 1]
        prior_factor[index + 1, index + 1] = sigma * np.sqrt(dt)
    prior_cov = prior_factor @ prior_factor.T
    posterior_cov = prior_cov - np.outer(prior_cov[:, -1], prior_cov[-1]) / (
        prior_cov[-1, -1] + tau**2
    )
    guide_factor = np.eye(len(steps) + 1)
    guide_factor[0, 0] = np.sqrt(posterior_cov[0, 0])
    for index, dt in enumerate(steps):
        guide_factor[index + 1, :index + 1] = (
            multiplier[index] / (1 + tilt[index]) * guide_factor[index, :index + 1]
        )
        guide_factor[index + 1, index + 1] = sigma * np.sqrt(dt)
    guide_cov = guide_factor @ guide_factor.T
    joint_kl = 0.5 * (
        np.trace(np.linalg.solve(posterior_cov, guide_cov)) - len(guide_cov)
        + np.linalg.slogdet(posterior_cov)[1] - np.linalg.slogdet(guide_cov)[1]
    )
    np.testing.assert_allclose(joint_kl, drift_only_gap, atol=1e-12)
    return dict(
        gap=gap, drift_only_gap=drift_only_gap, joint_kl=joint_kl,
        mean=mean, raw_mean=raw.mean(), raw_var=raw.var(ddof=1),
        expected_var=expected_var, first_var=debiased1.var(ddof=1),
        expected_first=expected_residual, max_oracle_error=np.max(np.abs(debiased2 - mean)),
    )


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for size in PILOT_SIZES:
        subset = [row for row in rows if row["pilot"] == size]
        for method in ("antithetic", "moment", "stein1", "stein2", "state_only"):
            values = np.array([row[method] for row in subset])
            lower, median, upper = np.quantile(values, [0.25, 0.50, 0.75])
            summary.append(dict(pilot=size, method=method, median=median, q25=lower, q75=upper))
    return summary


def local_gain(gradient, increments):
    """Estimate additive trace reduction using only held-out paths."""
    centered_gradient = gradient - gradient.mean(axis=0)
    centered_increments = increments - increments.mean(axis=0)
    cross = np.einsum("bp,bnp->n", centered_gradient, centered_increments)
    energy = np.einsum("bnp,bnp->n", centered_increments, centered_increments)
    return (2 * cross - energy) / (len(gradient) - 1)


def grid_benchmark():
    """Use nested Brownian increments for a grid and localization ablation."""
    rows = []
    refinements = (2, 4, 8)
    for case, times, y in cases():
        intervals = len(times)
        for seed in SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([5097, case, seed]))
            fine_pilot = rng.standard_normal((1024, intervals, refinements[-1]))
            fine_test = rng.standard_normal((4096, intervals, refinements[-1]))
            for substeps in refinements:
                stride = refinements[-1] // substeps

                def coarsen(noise):
                    return (noise.reshape(len(noise), -1, stride).sum(axis=-1)
                            / np.sqrt(stride))

                pilot = coarsen(fine_pilot)
                noise = coarsen(fine_test)
                _, _, pilot_state, mixed = nonlinear_paths(pilot, times, y, substeps=substeps)
                _, gradient, test_state, _ = nonlinear_paths(noise, times, y, substeps=substeps)
                targets = [mixed, mixed * pilot[:, :, None] / np.sqrt(2)]
                test_basis = basis(noise)
                base = variance(gradient)
                emit = schedule(times, y, substeps=substeps)[2]
                for method, tangent in (("tangent", True), ("state", False)):
                    increments = correction(
                        pilot_state, test_state, targets, test_basis, tangent, stepwise=True,
                    )
                    full_ratio = variance(gradient - increments.sum(axis=1)) / base
                    # Equal-sized subsets: the first and last step of each observation interval.
                    first = increments[:, np.roll(emit, 1), :].sum(axis=1)
                    last = increments[:, emit, :].sum(axis=1)
                    gain = local_gain(gradient, increments) / base
                    rows.append(dict(
                        case=case, seed=seed, substeps=substeps, method=method,
                        full_ratio=full_ratio, first_ratio=variance(gradient - first) / base,
                        last_ratio=variance(gradient - last) / base,
                        first_gain=gain[np.roll(emit, 1)].sum(), last_gain=gain[emit].sum(),
                        total_local_gain=gain.sum(),
                    ))
    return rows


def summarize_grid(rows):
    summary = []
    for substeps in (2, 4, 8):
        for method in ("tangent", "state"):
            selected = [row for row in rows
                        if row["substeps"] == substeps and row["method"] == method]
            center = {name: np.median([row[name] for row in selected]) for name in
                      ("full_ratio", "first_ratio", "last_ratio", "first_gain", "last_gain")}
            difference = [row["first_ratio"] - row["last_ratio"] for row in selected]
            lower, median, upper = np.quantile(difference, [0.25, 0.50, 0.75])
            summary.append(dict(
                substeps=substeps, method=method, **center,
                median_paired_improvement=median, paired_q25=lower, paired_q75=upper,
                last_wins=sum(value > 0 for value in difference), runs=len(selected),
            ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=Path("results"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    oracle = ou_benchmark()
    nonlinear = nonlinear_benchmark()
    summary = summarize(nonlinear)
    grid = grid_benchmark()
    grid_summary = summarize_grid(grid)
    contexts = [
        dict(case=case, time=time, observation=observation)
        for case, times, values in cases() for time, observation in zip(times, values)
    ]
    write_csv(args.out / "ou.csv", [oracle])
    write_csv(args.out / "nonlinear_runs.csv", nonlinear)
    write_csv(args.out / "nonlinear_summary.csv", summary)
    write_csv(args.out / "contexts.csv", contexts)
    write_csv(args.out / "grid_runs.csv", grid)
    write_csv(args.out / "grid_summary.csv", grid_summary)
    for row in [oracle, *summary, *grid_summary]:
        print("  ".join(f"{key}={value:.6g}" if isinstance(value, float)
                        else f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()
