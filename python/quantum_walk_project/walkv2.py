import argparse
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

import quantum_walk_project.walkers as qwp
qwp = qwp.walkersv2


@dataclass
class SimulationParameters:
    h_x: float = 0.25
    h_tau: float = 0.0625
    num_walkers: int = 10_000
    max_steps: int = 80
    max_tau: Optional[float] = None
    potential: int = 1  # 1=0.5x^2, 2=x


@dataclass
class SimulationResults:
    survival_counts: np.ndarray
    e0_estimates: np.ndarray
    e0_estimates_no_ln: np.ndarray
    active_walkers: np.ndarray
    final_walkers: np.ndarray
    active_walkers_at_all_steps: List[np.ndarray]
    params: SimulationParameters

    def get_tau_values(self) -> np.ndarray:
        """Convert step indices to tau values"""
        return np.arange(len(self.survival_counts)) * self.params.h_tau

    def get_survival_probabilities(self) -> np.ndarray:
        """Calculate survival probabilities"""
        return self.survival_counts / self.params.num_walkers


def parse_arguments() -> SimulationParameters:
    """Parse command line arguments and return parameters"""
    parser = argparse.ArgumentParser(
        description="Quantum Walker Simulation (v2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-h_x', '--h_x', type=float, default=0.25,
                        help='Step size in x direction')
    parser.add_argument('-nw', '--num_walkers', type=int,
                        default=10_000, help='Number of walkers')
    parser.add_argument('-ms', '--max_steps', type=int, default=96,
                        help='Maximum number of steps per walk 80')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the results')
    parser.add_argument('-mt', '--max_tau', type=float,
                        help='Maximum of tau (auto-calculates and overrides max_steps)')
    parser.add_argument('-pot', '--potential', type=int, default=1,
                        help='Potential type: 1=0.5x^2, 2=x')
    parser.add_argument('-a', '--auto_analyze', action='store_true',
                        help='Automatically analyze optimal parameters')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 0.1.0')

    args = parser.parse_args()

    h_tau = args.h_x ** 2

    params = SimulationParameters(
        h_x=args.h_x,
        h_tau=h_tau,
        num_walkers=args.num_walkers,
        max_steps=args.max_steps,
        max_tau=args.max_tau,
        potential=args.potential
    )

    if params.max_tau:
        params.max_steps = int(params.max_tau / params.h_tau)

    return params, args.plot, args.auto_analyze


def calculate_energy_errors(results: SimulationResults, theoretical_value: float, tau_value: float):
    """
    Calculate standard errors and confidence intervals for energy estimates at a given tau.
    Uses bootstrap resampling for more robust error estimation.
    """
    tau_values = results.get_tau_values()

    tau_idx = np.argmin(np.abs(tau_values - tau_value))
    actual_tau = tau_values[tau_idx]

    e0_estimate = results.e0_estimates[tau_idx]

    residual = e0_estimate - theoretical_value

    window_size = 5
    window_start = max(0, tau_idx - window_size)
    window_end = min(len(tau_values), tau_idx + window_size + 1)
    window_estimates = results.e0_estimates_no_ln[window_start:window_end]

    std_dev = np.std(window_estimates)

    sem = std_dev / np.sqrt(results.active_walkers[tau_idx])

    ci_lower = e0_estimate - 1.96 * sem
    ci_upper = e0_estimate + 1.96 * sem

    rel_error = abs(residual / theoretical_value) * 100

    return {
        'tau': actual_tau,
        'e0_estimate': e0_estimate,
        'std_dev': std_dev,
        'sem': sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'residual': residual,
        'rel_error': rel_error,
        'active_walkers': results.active_walkers[tau_idx]
    }


def bootstrap_energy_error(results: SimulationResults, tau_value: float, n_bootstrap=1000):
    """
    Perform bootstrap resampling to estimate error on energy at a given tau.
    """
    tau_values = results.get_tau_values()
    tau_idx = np.argmin(np.abs(tau_values - tau_value))
    actual_tau = tau_values[tau_idx]

    if tau_idx >= len(results.active_walkers_at_all_steps):
        print(f"Warning: Requested tau={tau_value} exceeds simulation time.")
        return None

    walkers = np.array(results.active_walkers_at_all_steps[tau_idx])
    if len(walkers) < 10:
        print(f"Warning: Too few walkers at tau={
              actual_tau} for bootstrap analysis.")
        return None

    if results.params.potential == 1:  # Harmonic oscillator
        potential_energies = 0.5 * walkers**2
    elif results.params.potential == 2:  # Linear potential
        potential_energies = np.abs(walkers)
    else:
        return None

    bootstrap_means = []
    n_walkers = len(walkers)

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_walkers, size=n_walkers)
        resampled_energies = potential_energies[indices]
        bootstrap_means.append(np.mean(resampled_energies))

    bootstrap_mean = np.mean(bootstrap_means)
    bootstrap_std = np.std(bootstrap_means)

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return {
        'tau': actual_tau,
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_std': bootstrap_std,
        'bootstrap_sem': bootstrap_std,
        'bootstrap_ci_lower': ci_lower,
        'bootstrap_ci_upper': ci_upper,
        'n_walkers': n_walkers
    }


def plot_energy_with_errors(results: SimulationResults, theoretical_value: float):
    """Plot energy estimates with error bars"""
    tau_values = results.get_tau_values()

    min_length = min(len(tau_values), len(results.e0_estimates))
    tau_values = tau_values[:min_length]
    e0_estimates = results.e0_estimates[:min_length]

    n_points = min(10, len(tau_values))
    sample_indices = np.linspace(0, len(tau_values)-1, n_points, dtype=int)

    error_bars = []
    for idx in sample_indices:
        tau = tau_values[idx]
        error_stats = calculate_energy_errors(results, theoretical_value, tau)
        error_bars.append(error_stats)

    plt.figure(figsize=(12, 7))

    plt.plot(tau_values, e0_estimates, 'b-', alpha=0.5, label='E₀ estimates')

    error_x = [stat['tau'] for stat in error_bars]
    error_y = [stat['e0_estimate'] for stat in error_bars]
    error_yerr = [stat['sem'] for stat in error_bars]

    plt.errorbar(error_x, error_y, yerr=error_yerr, fmt='ro',
                 capsize=4, label='E₀ with standard error')

    plt.axhline(y=theoretical_value, color='g', linestyle='--',
                label=f'Theoretical: {theoretical_value:.6f}')

    for stat in error_bars:
        plt.fill_between([stat['tau']-0.05, stat['tau']+0.05],
                         [stat['ci_lower'], stat['ci_lower']],
                         [stat['ci_upper'], stat['ci_upper']],
                         color='r', alpha=0.2)

    plt.xlabel('Time (τ)')
    plt.ylabel('Ground State Energy (E₀)')
    plt.title('Ground State Energy Estimates with Error Bars')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_convergence(results: SimulationResults, theoretical_value: float):
    """Plot how the error in energy estimates converges with increasing tau"""
    tau_values = results.get_tau_values()

    min_length = min(len(tau_values), len(results.e0_estimates_no_ln))
    tau_values = tau_values[:min_length]
    e0_estimates = results.e0_estimates_no_ln[:min_length]

    abs_errors = np.abs(e0_estimates - theoretical_value)

    rel_errors = abs_errors / theoretical_value * 100

    n_points = min(10, len(tau_values))
    sample_indices = np.linspace(0, len(tau_values)-1, n_points, dtype=int)

    std_errors = []
    for idx in sample_indices:
        tau = tau_values[idx]
        error_stats = calculate_energy_errors(results, theoretical_value, tau)
        std_errors.append(error_stats['sem'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot absolute error
    ax1.semilogy(tau_values, abs_errors, 'b-', label='Absolute Error')
    ax1.set_xlabel('Time (τ)')
    ax1.set_ylabel('Absolute Error |E₀ - E₀_theoretical|')
    ax1.set_title('Absolute Error Convergence')
    ax1.grid(True, alpha=0.3, which='both')

    # Plot relative error
    ax2.semilogy(tau_values, rel_errors, 'r-', label='Relative Error (%)')
    ax2.set_xlabel('Time (τ)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('Relative Error Convergence')
    ax2.grid(True, alpha=0.3, which='both')

    error_x = [tau_values[idx] for idx in sample_indices]
    ax1.scatter(error_x, std_errors, color='g', marker='x',
                label='Standard Error')

    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.show()


def print_statistics_with_errors(results: SimulationResults, theoretical_value: float, tau_value: float):
    """Print detailed statistics with error analysis at a given tau"""
    error_stats = calculate_energy_errors(
        results, theoretical_value, tau_value)

    bootstrap_stats = bootstrap_energy_error(results, tau_value)

    print(f"\nDetailed Statistics at τ = {error_stats['tau']:.4f}:")
    print(f"  Ground State Energy Estimate: {error_stats['e0_estimate']:.6f}")
    print(f"  Theoretical Value: {theoretical_value:.6f}")
    print(f"  Absolute Error: {abs(error_stats['residual']):.6f}")
    print(f"  Relative Error: {error_stats['rel_error']:.2f}%")
    print(f"  Standard Error: {error_stats['sem']:.6f}")
    print(f"  95% Confidence Interval: [{
          error_stats['ci_lower']:.6f}, {error_stats['ci_upper']:.6f}]")

    if error_stats['ci_lower'] <= theoretical_value <= error_stats['ci_upper']:
        print("  ✅ Theoretical value is within the 95% confidence interval")
    else:
        print("  ❌ Theoretical value is outside the 95% confidence interval")

    print(f"  Number of active walkers: {error_stats['active_walkers']}")

    if bootstrap_stats:
        print("\n  Bootstrap Analysis:")
        print(f"    Bootstrap Mean: {bootstrap_stats['bootstrap_mean']:.6f}")
        print(f"    Bootstrap Standard Error: {
              bootstrap_stats['bootstrap_sem']:.6f}")
        print(f"    Bootstrap 95% CI: [{bootstrap_stats['bootstrap_ci_lower']:.6f}, "
              f"{bootstrap_stats['bootstrap_ci_upper']:.6f}]")

        if (bootstrap_stats['bootstrap_ci_lower'] <= theoretical_value <=
                bootstrap_stats['bootstrap_ci_upper']):
            print("    ✅ Theoretical value is within the bootstrap confidence interval")
        else:
            print("    ❌ Theoretical value is outside the bootstrap confidence interval")


def run_simulation(params: SimulationParameters) -> SimulationResults:
    """Run the quantum walk simulation with given parameters"""
    survival_counts, e0_estimates, e0_estimates_no_ln, active_walkers, final_walkers, active_walkers_at_all_steps = (
        qwp.run_walkv2(
            params.h_x,
            params.h_tau,
            params.num_walkers,
            params.max_steps,
            params.potential
        )
    )

    return SimulationResults(
        survival_counts=np.array(survival_counts),
        e0_estimates=np.array(e0_estimates),
        e0_estimates_no_ln=np.array(e0_estimates_no_ln),
        active_walkers=np.array(active_walkers),
        final_walkers=np.array(final_walkers),
        active_walkers_at_all_steps=active_walkers_at_all_steps,
        params=params
    )


def find_optimal_tau(results: SimulationResults, theoretical_value: float) -> float:
    """Find the optimal tau value where energy estimate stabilizes"""
    tau_values = results.get_tau_values()

    min_length = min(len(tau_values), len(results.e0_estimates_no_ln))
    tau_values = tau_values[:min_length]
    e0_estimates = results.e0_estimates_no_ln[:min_length]

    residuals = np.abs(e0_estimates - theoretical_value)

    window_size = min(10, len(residuals) // 5) if len(residuals) > 20 else 3
    if len(residuals) < window_size:
        return tau_values[-1] if len(tau_values) > 0 else 2.0

    windowed_avg = np.convolve(residuals, np.ones(
        window_size)/window_size, mode='valid')

    stable_threshold = np.mean(windowed_avg) * 0.8
    stable_indices = np.where(windowed_avg < stable_threshold)[0]

    if len(stable_indices) > 0:
        optimal_idx = stable_indices[0] + window_size // 2  # Adjust for window
        optimal_tau = tau_values[optimal_idx] if optimal_idx < len(
            tau_values) else tau_values[-1]
    else:
        optimal_idx = np.argmin(residuals)
        optimal_tau = tau_values[optimal_idx] if optimal_idx < len(
            tau_values) else tau_values[-1]

    return optimal_tau


def auto_analyze(theoretical_value: float) -> SimulationParameters:
    """
    Perform parameter sweeping to find optimal simulation parameters
    based on convergence to the theoretical ground state energy.
    """
    print("Starting auto-analysis to find optimal parameters...")

    h_x_values = [0.1, 0.2, 0.25, 0.3, 0.5]
    h_tau_values = [0.01, 0.03125, 0.0625, 0.125, 0.25]
    num_walkers_values = [1000, 5000, 10000]

    best_residual = float('inf')
    best_params = None
    results_data = []

    for h_x in h_x_values:
        for h_tau in h_tau_values:
            for num_walkers in num_walkers_values:
                sweep_params = SimulationParameters(
                    h_x=h_x,
                    h_tau=h_tau,
                    num_walkers=num_walkers,
                    max_steps=min(80, int(5 / h_tau))  # Go up to tau=5
                )

                results = run_simulation(sweep_params)

                optimal_tau = find_optimal_tau(results, theoretical_value)
                optimal_step = int(optimal_tau / h_tau)

                if optimal_step < len(results.e0_estimates_no_ln):
                    e0_estimate = results.e0_estimates_no_ln[optimal_step]
                    residual = abs(e0_estimate - theoretical_value)

                    results_data.append({
                        'h_x': h_x,
                        'h_tau': h_tau,
                        'num_walkers': num_walkers,
                        'optimal_tau': optimal_tau,
                        'e0_estimate': e0_estimate,
                        'residual': residual
                    })

                    if residual < best_residual:
                        best_residual = residual
                        best_params = SimulationParameters(
                            h_x=h_x,
                            h_tau=h_tau,
                            num_walkers=num_walkers,
                            max_steps=min(200, int(10 / h_tau)),
                            potential=1
                        )

    results_array = np.array([(d['h_x'], d['h_tau'], d['num_walkers'], d['residual'])
                              for d in results_data],
                             dtype=[('h_x', float), ('h_tau', float),
                                    ('num_walkers', int), ('residual', float)])

    good_threshold = best_residual * 1.2
    good_results = [r for r in results_data if r['residual'] <= good_threshold]

    if best_params:
        print("\nAuto-analysis results:")
        print("Best parameters found:")
        print(f"  h_x: {best_params.h_x}")
        print(f"  h_tau: {best_params.h_tau}")
        print(f"  num_walkers: {best_params.num_walkers}")
        print(f"  residual: {best_residual:.6f}")

        if len(good_results) > 1:
            sorted_good = sorted(good_results,
                                 key=lambda r: (r['num_walkers'], -r['h_x'], -r['h_tau']))
            efficient = sorted_good[0]

            print("\nMost efficient parameters with good accuracy:")
            print(f"  h_x: {efficient['h_x']}")
            print(f"  h_tau: {efficient['h_tau']}")
            print(f"  num_walkers: {efficient['num_walkers']}")
            print(f"  residual: {efficient['residual']:.6f}")

    return best_params or SimulationParameters()


def plot_survival_and_energy(results: SimulationResults):
    """Plot survival probability and energy estimates"""
    tau_values = results.get_tau_values()
    survival_prob = results.get_survival_probabilities()

    min_length = min(len(tau_values), len(results.e0_estimates))
    tau_values_plot = tau_values[:min_length]
    e0_estimates_plot = results.e0_estimates[:min_length]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(tau_values[:len(survival_prob)], survival_prob,
             label='Survival probability', color='tab:blue')
    ax1.set_xlabel('Time (τ)')
    ax1.set_ylabel('Survival probability', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.scatter(tau_values_plot, e0_estimates_plot, label='E₀ estimate',
                color='tab:red', marker='x', s=15, alpha=0.7)
    ax2.set_ylabel('E₀ estimate', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('Survival Probability and Ground State Energy Estimates')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_energy_residuals(results: SimulationResults, theoretical_value: float):
    """Plot residuals of energy estimates from theoretical value"""
    tau_values = results.get_tau_values()

    min_length = min(len(tau_values), len(results.e0_estimates_no_ln))
    tau_values_plot = tau_values[:min_length]
    residuals = results.e0_estimates_no_ln[:min_length] - theoretical_value

    plt.figure(figsize=(10, 6))
    plt.plot(tau_values_plot, residuals, marker='o', markersize=4, alpha=0.7)

    plt.xlabel('Time (τ)')
    plt.ylabel('Residual (E₀ estimate - theoretical)')
    plt.title('Residuals of Ground State Energy Estimates')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_wave_function(results: SimulationResults, tau_value: float):
    """Plot ground state wave function at given tau"""
    target_step = int(tau_value / results.params.h_tau)

    if target_step >= len(results.active_walkers_at_all_steps):
        print(f"Warning: Requested tau={
              tau_value} exceeds simulation time. Using maximum available.")
        target_step = len(results.active_walkers_at_all_steps) - 1

    walkers_at_tau = np.array(results.active_walkers_at_all_steps[target_step])

    if len(walkers_at_tau) == 0:
        print(f"Error: No walkers available at tau={tau_value}")
        return

    h_x = results.params.h_x
    bins = np.arange(-3, 3, 2*h_x)

    wave_func = np.zeros(len(bins))
    for i, bin_edge in enumerate(bins):
        values_in_bin = walkers_at_tau[
            (walkers_at_tau >= bin_edge) &
            (walkers_at_tau < bin_edge + 2*h_x)
        ]
        wave_func[i] = len(values_in_bin) / (len(walkers_at_tau) * 2*h_x)

    norm_sum = np.sum(wave_func * 2*h_x)
    if not math.isclose(norm_sum, 1, abs_tol=1e-2):
        print(f"Warning: Sum of probabilities is {norm_sum}, not 1.")

    plt.figure(figsize=(10, 6))
    plt.bar(bins, wave_func, width=2*h_x, align='edge', alpha=0.6,
            label=f'Simulation (τ={tau_value})')

    x = np.linspace(-3, 3, 300)
    if results.params.potential == 1:  # Harmonic oscillator
        psi_0 = np.pi ** -0.25 * np.exp(-x**2/2)
        plt.plot(x, psi_0, color='red', label='Exact ψ₀(x)')
    elif results.params.potential == 2:  # Linear potential
        pass

    plt.xlabel('Position (x)')
    plt.ylabel('Wave function ψ₀(x)')
    plt.title(f'Ground State Wave Function at τ={tau_value}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_probability_density(results: SimulationResults, tau_value: float):
    """Plot probability density |ψ₀(x)|² at given tau"""
    target_step = int(tau_value / results.params.h_tau)

    if target_step >= len(results.active_walkers_at_all_steps):
        print(f"Warning: Requested tau={
              tau_value} exceeds simulation time. Using maximum available.")
        target_step = len(results.active_walkers_at_all_steps) - 1

    walkers_at_tau = np.array(results.active_walkers_at_all_steps[target_step])

    if len(walkers_at_tau) == 0:
        print(f"Error: No walkers available at tau={tau_value}")
        return

    h_x = results.params.h_x
    bins = np.arange(-3, 3, 2*h_x)

    prob_density = np.zeros(len(bins))
    for i, bin_edge in enumerate(bins):
        values_in_bin = walkers_at_tau[
            (walkers_at_tau >= bin_edge) &
            (walkers_at_tau < bin_edge + 2*h_x)
        ]
        prob_density[i] = len(values_in_bin) / (len(walkers_at_tau) * 2*h_x)

    norm_sum = np.sum(prob_density * 2*h_x)
    if not math.isclose(norm_sum, 1, abs_tol=1e-2):
        print(f"Warning: Sum of probabilities is {norm_sum}, not 1.")

    plt.figure(figsize=(10, 6))
    plt.bar(bins, prob_density, width=2*h_x, align='edge', alpha=0.6,
            label=f'Simulation (τ={tau_value})')

    x = np.linspace(-3, 3, 300)
    if results.params.potential == 1:  # Harmonic oscillator
        psi_0_sq = np.pi ** -0.5 * np.exp(-x**2)
        plt.plot(x, psi_0_sq, color='red', label='Exact |ψ₀(x)|²')
    elif results.params.potential == 2:  # Linear potential
        pass

    plt.xlabel('Position (x)')
    plt.ylabel('Probability Density |ψ₀(x)|²')
    plt.title(f'Ground State Probability Density at τ={tau_value}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_statistics(results: SimulationResults, theoretical_value: float, optimal_tau: float):
    """Print statistics about the simulation results"""
    tau_values = results.get_tau_values()

    optimal_idx = np.argmin(np.abs(tau_values - optimal_tau))
    e0_at_optimal = results.e0_estimates[optimal_idx]
    e0_no_ln_at_optimal = results.e0_estimates_no_ln[optimal_idx]
    residual = e0_no_ln_at_optimal - theoretical_value

    window_start = max(0, optimal_idx - 5)
    window_end = min(len(tau_values), optimal_idx + 6)
    window_estimates = results.e0_estimates_no_ln[window_start:window_end]

    mean_e0 = np.mean(window_estimates)
    std_e0 = np.std(window_estimates)

    print("Simulation Statistics:")
    print(f"\tOptimal tau: {optimal_tau:.4f}")
    print(f"\tTheoretical ground state energy: {theoretical_value:.6f}")
    print(f"\tE₀ estimate (ratio) at τ={optimal_tau:.2f}: {e0_at_optimal:.6f}")
    print(f"\tE₀ estimate (without ln) at τ={
          optimal_tau:.2f}: {e0_no_ln_at_optimal:.6f}")
    print(f"\tResidual at τ={optimal_tau:.2f}: {residual:.6f}")
    print(f"\tMean E₀ estimate near τ={optimal_tau:.2f}: {mean_e0:.6f}")
    print(f"\tStandard deviation of E₀ near τ={optimal_tau:.2f}: {std_e0:.6f}")

    later_idx = min(len(tau_values) - 1, optimal_idx + 10)
    if later_idx > optimal_idx:
        later_residual = abs(
            results.e0_estimates_no_ln[later_idx] - theoretical_value)
        if later_residual < abs(residual):
            print(f"Note: Energy estimate continues to improve after τ={
                  optimal_tau:.2f}")


def main():
    params, do_plot, do_auto_analyze = parse_arguments()

    if params.potential == 1:  # Harmonic oscillator
        theoretical_value = 0.5  # Ground state energy of harmonic oscillator
    elif params.potential == 2:  # Linear potential
        theoretical_value = 0.808614  # First zero of Airy function
    else:
        theoretical_value = 0.0
        print(f"Warning: Unknown potential type {params.potential}")

    if do_auto_analyze:
        optimal_params = auto_analyze(theoretical_value)
        params = optimal_params

    results = run_simulation(params)

    optimal_tau = find_optimal_tau(results, theoretical_value)

    if do_plot:
        plot_survival_and_energy(results)
        plot_energy_residuals(results, theoretical_value)
        plot_wave_function(results, 2.0)  # Plot at tau=2
        plot_probability_density(results, 4.0)  # Plot at tau=4

        plot_energy_with_errors(results, theoretical_value)
        plot_error_convergence(results, theoretical_value)

    print_statistics_with_errors(results, theoretical_value, optimal_tau)
    print_statistics_with_errors(results, theoretical_value, 2.0)
    print_statistics_with_errors(results, theoretical_value, 4.0)
