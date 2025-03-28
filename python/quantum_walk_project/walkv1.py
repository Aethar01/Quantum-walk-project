import argparse
import math
from typing import Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

import quantum_walk_project.walkers as qwp
qwp = qwp.walkersv1


@dataclass
class SimulationParameters:
    j_range: Tuple[int, int, int]
    walker_range: Tuple[int, int, int]
    step_range: Tuple[int, int, int]
    seed: int = 0
    threads: int = 0
    runs: int = 1


def parse_arguments():
    parser = argparse.ArgumentParser(
            description="Quantum Walker Simulation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # Walker parameters
    walker_group = parser.add_argument_group("Walker Parameters")
    walker_group.add_argument('-nu', '--num_walkers_upper', type=int, default=100,
                              help='Maximum number of random walks')
    walker_group.add_argument('-nl', '--num_walkers_lower', type=int, default=100,
                              help='Minimum number of random walks')
    walker_group.add_argument('-ns', '--num_walkers_step', type=int, default=100,
                              help='Step size for number of random walks')

    # Lattice parameters
    lattice_group = parser.add_argument_group("Lattice Parameters")
    lattice_group.add_argument('-ju', '--j_upper', type=int, default=10,
                               help='Maximum lattice size J (boundary at ±J)')
    lattice_group.add_argument('-jl', '--j_lower', type=int, default=10,
                               help='Minimum lattice size J (boundary at ±J)')
    lattice_group.add_argument('-js', '--j_step', type=int, default=2,
                               help='Step size for lattice size')

    # Step parameters
    step_group = parser.add_argument_group("Step Parameters")
    step_group.add_argument('-mu', '--max_steps_upper', type=int, default=10000,
                            help='Maximum number of steps per walk')
    step_group.add_argument('-ml', '--max_steps_lower', type=int, default=100,
                            help='Minimum number of steps per walk')
    step_group.add_argument('-ms', '--max_steps_step', type=int, default=100,
                            help='Step size for number of steps per walk')

    # Other parameters
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Seed for RNG. If 0, seed is random. NOTE: With seed, process is not multithreaded')
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help='Number of threads to use. If 0, use all available threads')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('-r', '--runs', type=int, default=5,
                        help='Number of times to run the simulation')
    parser.add_argument('-a', '--analyse', action='store_true',
                        help='Narrow down parameters automatically')
    parser.add_argument('-ar', '--analyse_runs', type=int, default=100,
                        help='Number of runs to analyse')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the results')
    parser.add_argument('-sd', '--save_dir', type=str, default=None, help='Directory to save plots')
    parser.add_argument('-pb', '--progressbar',
                        action='store_true', help='Display progress bar')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 0.1.0')

    return parser.parse_args()


def run_simulations(params: SimulationParameters, progressbar: bool) -> pd.DataFrame:
    """Run quantum walker simulations with the given parameters and return results dataframe."""
    j_lower, j_step, j_upper = params.j_range
    walker_lower, walker_step, walker_upper = params.walker_range
    step_lower, step_step, step_upper = params.step_range

    results = []

    for _ in range(params.runs):
        batch_results = qwp.run_many_run_walk(
            j_lower, j_step, j_upper,
            walker_lower, walker_step, walker_upper,
            step_lower, step_step, step_upper,
            threads=params.threads, seed=params.seed,
            quiet=not progressbar
        )

        for r in batch_results:
            results.append({
                'j': r.j,
                'num_walkers': r.num_walkers,
                'max_steps': r.max_steps,
                'j_sq': r.j_sq,
                'lambda_j_sq': r.lambda_j_sq,
                'residual': r.residual
            })

    return pd.DataFrame(results)


def auto_analyse(df: pd.DataFrame, theoretical: float) -> Tuple[SimulationParameters, pd.DataFrame]:
    """
    Perform intelligent analysis of the simulation results to find optimal parameters.
    Returns refined parameters and a filtered dataframe with best results.
    """
    min_residual = df['residual'].min()
    threshold = min_residual * 1.1
    good_results = df[df['residual'] <= threshold]

    j_values = good_results['j'].value_counts()
    walker_values = good_results['num_walkers'].value_counts()
    step_values = good_results['max_steps'].value_counts()

    best_j = j_values.idxmax()
    best_walkers = walker_values.idxmax()
    best_steps = step_values.idxmax()

    j_candidates = j_values[j_values > len(
        good_results) / len(j_values)].index.tolist()
    walker_candidates = walker_values[walker_values > len(
        good_results) / len(walker_values)].index.tolist()
    step_candidates = step_values[step_values > len(
        good_results) / len(step_values)].index.tolist()

    if len(j_candidates) > 1:
        j_range = (min(j_candidates), 2, max(j_candidates))
    else:
        j_range = (best_j, 1, best_j)

    if len(walker_candidates) > 1:
        walker_range = (min(walker_candidates), 100, max(walker_candidates))
    else:
        walker_range = (best_walkers, 1, best_walkers)

    if len(step_candidates) > 1:
        step_range = (min(step_candidates), 100, max(step_candidates))
    else:
        step_range = (best_steps, 1, best_steps)

    print("Auto-analysis results:")
    print(f"Optimal J range: {j_range}")
    print(f"Optimal walker count range: {walker_range}")
    print(f"Optimal step count range: {step_range}")
    print(f"Best residual: {min_residual:.6f}")

    params = SimulationParameters(
        j_range=j_range,
        walker_range=walker_range,
        step_range=step_range
    )

    return params, good_results


def plot_results(df: pd.DataFrame, theoretical: float, args):
    """Generate plots to visualize the simulation results."""
    fig, ax = plt.subplots(2, 2, figsize=(4, 3))
    fig.suptitle('Parameter Impact on Residual')

    # J vs Residual
    sns_plot1 = ax[0, 0]
    sns_plot1.scatter(df['j'], df['residual'], alpha=0.7)
    sns_plot1.set_xlabel('L/2 (Lattice Size)')
    sns_plot1.set_ylabel('Residual')
    sns_plot1.grid(True, alpha=0.3)

    # Number of Walkers vs Residual
    sns_plot2 = ax[0, 1]
    sns_plot2.scatter(df['num_walkers'], df['residual'], alpha=0.7)
    sns_plot2.set_xlabel('Number of Walkers')
    sns_plot2.set_ylabel('Residual')
    sns_plot2.grid(True, alpha=0.3)

    # Max Steps vs Residual
    sns_plot3 = ax[1, 0]
    sns_plot3.scatter(df['max_steps'], df['residual'], alpha=0.7)
    sns_plot3.set_xlabel('Max Steps')
    sns_plot3.set_ylabel('Residual')
    sns_plot3.grid(True, alpha=0.3)

    # L/2² vs Residual
    sns_plot4 = ax[1, 1]
    sns_plot4.scatter(df['j_sq'], df['residual'], alpha=0.7)
    sns_plot4.set_xlabel('$L/2^2$')
    sns_plot4.set_ylabel('Residual')
    sns_plot4.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.plot:
        plt.show()

    if args.save_dir:
        plt.savefig(f'{args.save_dir}/analysis.pgf')
    plt.close()

    # Lambda L/2² histogram with theoretical value and error bars
    plt.figure(figsize=(4, 3))

    error_stats = calculate_errors(df, theoretical)
    mean_value = error_stats['mean']

    plt.hist(df['lambda_j_sq'], bins=50, alpha=0.7, label='Simulated $\\lambda (L/2)^2$')
    plt.axvline(x=theoretical, color='red', linestyle='--',
                linewidth=1, label=f'Theoretical: $\\pi^2/8$ = {theoretical:.6f}')
    plt.axvline(x=mean_value, color='green', linestyle='-',
                linewidth=1, label=f'Mean: {mean_value:.6f}')

    plt.axvspan(error_stats['ci_lower'], error_stats['ci_upper'],
                alpha=0.2, color='green', label='95% Confidence Interval')

    plt.xlabel('$\\lambda (L/2)^2$')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.title('Distribution of λL/2² Values with Error Analysis')
    if args.save_dir:
        plt.savefig(f'{args.save_dir}/histogram.pgf')
    if args.plot:
        plt.show()
    plt.close()

    if len(df) > 10:  # Only if we have enough data points
        plt.figure(figsize=(4, 3))

        df_sorted = df.sort_values('num_walkers')

        running_means = []
        running_ci_lower = []
        running_ci_upper = []

        for i in range(1, len(df_sorted) + 1):
            subset = df_sorted.iloc[:i]
            stats = calculate_errors(subset, theoretical)
            running_means.append(stats['mean'])
            running_ci_lower.append(stats['ci_lower'])
            running_ci_upper.append(stats['ci_upper'])

        x = range(1, len(df_sorted) + 1)
        plt.plot(x, running_means, 'b-', label='Running Mean', markersize=1)
        plt.fill_between(x, running_ci_lower, running_ci_upper,
                         color='b', alpha=0.2, label='95% CI')
        plt.axhline(y=theoretical, color='r', linestyle='--',
                    label=f'Theoretical: {theoretical:.6f}', linewidth=1)

        plt.xlabel('Number of Samples')
        plt.ylabel('$\\lambda (L/2)^2$')
        # plt.title('Convergence of λL/2² with Increasing Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if args.save_dir:
            plt.savefig(f'{args.save_dir}/convergence.pgf')
        if args.plot:
            plt.show()


def calculate_bootstrap_ci(data, confidence=0.95, n_resamples=10000):
    """Calculate bootstrap confidence intervals."""
    from numpy.random import choice

    bootstrap_means = []
    n = len(data)

    for _ in range(n_resamples):
        sample = choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrap_means, 100 * alpha)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha))

    return lower_bound, upper_bound


def calculate_errors(df: pd.DataFrame, theoretical: float):
    """Calculate standard errors and confidence intervals for energy measurements."""
    n_samples = len(df)

    mean_lambda_j_sq = df['lambda_j_sq'].mean()
    std_dev = df['lambda_j_sq'].std()

    sem = std_dev / np.sqrt(n_samples)

    from scipy import stats
    confidence = 0.95
    t_critical = stats.t.ppf((1 + confidence) / 2, n_samples - 1)
    ci_half_width = t_critical * sem
    ci_lower = mean_lambda_j_sq - ci_half_width
    ci_upper = mean_lambda_j_sq + ci_half_width

    residual = abs(theoretical - mean_lambda_j_sq)

    relative_error = (residual / theoretical) * 100

    residual_error = sem

    return {
        'mean': mean_lambda_j_sq,
        'std_dev': std_dev,
        'sem': sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'residual': residual,
        'residual_error': residual_error,
        'relative_error': relative_error
    }


def print_statistics(df: pd.DataFrame, theoretical: float):
    """Calculate and print key statistics from the simulation results."""
    average = df['lambda_j_sq'].mean()
    median = df['lambda_j_sq'].median()

    mode_resolution = 1e-3
    bins = np.arange(df['lambda_j_sq'].min(),
                     df['lambda_j_sq'].max() + mode_resolution,
                     mode_resolution)
    hist, _ = np.histogram(df['lambda_j_sq'], bins)
    mode = bins[hist.argmax()]

    std_dev = df['lambda_j_sq'].std()
    min_val = df['lambda_j_sq'].min()
    max_val = df['lambda_j_sq'].max()

    error_stats = calculate_errors(df, theoretical)

    print("\nSimulation Statistics:")
    print(f"Theoretical Value (π²/8): {theoretical:.6f}")
    print(f"Average λJ²: {average:.6f} (residual: {
          abs(theoretical - average):.6f})")
    print(f"Median λJ²: {median:.6f} (residual: {
          abs(theoretical - median):.6f})")
    print(f"Mode λJ²: {mode:.6f} (residual: {abs(theoretical - mode):.6f})")
    print(f"Standard Deviation: {std_dev:.6f}")
    print(f"Range: [{min_val:.6f}, {max_val:.6f}]")

    print("\nError Analysis:")
    print(f"Standard Error of Mean (SEM): {error_stats['sem']:.6f}")
    print(f"Residual: {error_stats['residual']:.6f} ± {
          error_stats['residual_error']:.6f}")
    print(f"Relative Error: {error_stats['relative_error']:.2f}%")
    print(f"95% Confidence Interval: [{error_stats['ci_lower']:.6f}, {
          error_stats['ci_upper']:.6f}]")

    if error_stats['ci_lower'] <= theoretical <= error_stats['ci_upper']:
        print("✅ Theoretical value is within the 95% confidence interval")
    else:
        print("❌ Theoretical value is outside the 95% confidence interval")

    bootstrap_ci = calculate_bootstrap_ci(
        df['lambda_j_sq'].values, confidence=0.95)
    print(f"\nBootstrap 95% CI: [{
          bootstrap_ci[0]:.6f}, {bootstrap_ci[1]:.6f}]")

    if bootstrap_ci[0] <= theoretical <= bootstrap_ci[1]:
        print("✅ Theoretical value is within the bootstrap confidence interval")
    else:
        print("❌ Theoretical value is outside the bootstrap confidence interval")


def main():
    args = parse_arguments()
    theoretical = (math.pi ** 2) / 8

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            print("Error: Directory does not exist.")
            print("Plots will be displayed only.")
            args.save_dir = None
        else:
            plt.rcParams.update({
                "text.usetex": True,
                "pgf.rcfonts": False,
                "pgf.texsystem": "pdflatex",
                "font.size": 8,
            })

    params = SimulationParameters(
        j_range=(args.j_lower, args.j_step, args.j_upper),
        walker_range=(args.num_walkers_lower,
                      args.num_walkers_step, args.num_walkers_upper),
        step_range=(args.max_steps_lower, args.max_steps_step,
                    args.max_steps_upper),
        seed=args.seed,
        threads=args.threads,
        runs=args.runs
    )

    df = run_simulations(params, True)

    if args.analyse:
        refined_params, good_results = auto_analyse(df, theoretical)
        refined_params.runs = args.analyse_runs
        refined_params.seed = args.seed
        refined_params.threads = args.threads

        df = run_simulations(refined_params, args.progressbar)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

    if args.plot or args.save_dir:
        plot_results(df, theoretical, args)

    print_statistics(df, theoretical)
