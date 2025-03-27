import argparse
import matplotlib.pyplot as plt
import numpy as np
import quantum_walk_project.walkers as qwp
from scipy import stats as statistics
import math
import os
import time
qwp = qwp.walkersv3


def parse_args():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n0', '--n0', type=int, default=50,
                        help='Initial number of walkers')
    parser.add_argument('-ms', '--max_steps', type=int, default=50000,
                        help='Number of Monte Carlo steps')
    parser.add_argument('-h_x', '--h_x', type=float,
                        default=0.1, help='Step length')
    parser.add_argument('-w0', '--w0', type=float, default=1.0,
                        help='Initial distribution width')
    parser.add_argument('-ss', '--save_steps', type=int, nargs='+',
                        default=[500, 50000], help='Steps at which to save histograms')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot results')
    parser.add_argument('-sd', '--save_dir', type=str, default=None,
                        help='Save plots to dir')
    parser.add_argument('-e', '--equilibration', type=int, default=1000,
                        help='Number of steps to discard')
    return parser.parse_args()


def calculate_histogram(walker_positions, xmin=-6.0, xmax=6.0, nbins=40):
    """Calculate histogram from walker positions"""
    hist, bin_edges = np.histogram(
        walker_positions, bins=nbins, range=(xmin, xmax), density=1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    return bin_centers, hist, bin_width


def collect_walkers_before_step(results, target_step):
    """Collect walker positions from steps up to the target step"""

    in_range_steps = [s for s in range(1, target_step+1)]

    all_walkers = []
    for step in in_range_steps:
        try:
            walkers = np.array(results[f'walkers_{step}'])
        except KeyError:
            continue
        all_walkers.extend(walkers.flatten())

    return np.array(all_walkers), step


def plot_wave_function(results, target_steps, args):
    """Plot wave function at specific steps"""
    xmin, xmax = -6.0, 6.0
    x_exact = np.linspace(xmin, xmax, 200)
    # psi_exact = np.pi**(-0.25) * np.exp(-0.5 * x_exact**2)
    psi_exact = (2*np.pi)**(-0.5) * np.exp(-0.5 * x_exact**2)

    plt.figure(figsize=(4, 3))

    plt.plot(x_exact, psi_exact, 'k-', alpha=0.5, label='Theoretical', linewidth=1)

    colors = ['bo--', 'gs', 'r+', 'mx:']
    for i, step in enumerate(target_steps):
        walkers, step = collect_walkers_before_step(results, step)

        if len(walkers) > 0:
            x_bins, hist, bin_width = calculate_histogram(walkers)
            norm_sum = np.sum(hist) * (x_bins[1] - x_bins[0])
            if not math.isclose(norm_sum, 1, abs_tol=1e-2):
                print(f"Warning: Sum of probabilities is {norm_sum}, not 1.")
            # plt.bar(x_bins, hist, width=bin_width, alpha=0.5, color=colors[i % len(
            #     colors)][0], label=f'Step {len(walkers)}')
            plt.plot(x_bins, hist, colors[i % len(colors)], label=f'Step {step}', linewidth=1, markersize=3, fillstyle='none')

    plt.xlabel('$x$')
    plt.ylabel('$\\psi(x)$')
    plt.legend()
    plt.grid(True)
    plt.xlim(-4, 4)
    # plt.ylim(0, 0.8)
    if args.save_dir:
        plt.savefig(f'{args.save_dir}/wave_function.pgf')
    if args.plot:
        plt.show()
    plt.close()


def calculate_energy_statistics(energy_data, equilibration=1000, theoretical_value=0.5):
    """Calculate energy statistics including error estimates"""
    equilibrated_data = energy_data[energy_data[:, 0] >= equilibration, 1]

    if len(equilibrated_data) == 0:
        print("Warning: No data points after discarded period")
        return None

    mean_energy = np.mean(equilibrated_data)
    std_dev = np.std(equilibrated_data)
    sem = std_dev / np.sqrt(len(equilibrated_data))

    ci_95 = statistics.t.interval(0.95, len(equilibrated_data)-1,
                                  loc=mean_energy,
                                  scale=sem)

    abs_error = abs(mean_energy - theoretical_value)
    rel_error = abs_error / theoretical_value * 100

    autocorr = calculate_autocorrelation(equilibrated_data)

    if autocorr > 0:
        effective_samples = len(equilibrated_data) / (2 * autocorr)
        corrected_sem = std_dev / np.sqrt(effective_samples)
    else:
        effective_samples = len(equilibrated_data)
        corrected_sem = sem

    corrected_ci_95 = statistics.t.interval(0.95, max(1, int(effective_samples)-1),
                                            loc=mean_energy,
                                            scale=corrected_sem)

    return {
        'mean': mean_energy,
        'std_dev': std_dev,
        'sem': sem,
        'ci_95': ci_95,
        'abs_error': abs_error,
        'rel_error': rel_error,
        'autocorr_time': autocorr,
        'effective_samples': effective_samples,
        'corrected_sem': corrected_sem,
        'corrected_ci_95': corrected_ci_95,
        'n_samples': len(equilibrated_data)
    }


def calculate_autocorrelation(data):
    """Calculate autocorrelation time of energy data"""
    data_norm = data - np.mean(data)

    acf = np.correlate(data_norm, data_norm, mode='full')
    acf = acf[len(data_norm)-1:] / (np.var(data_norm)
                                    * np.arange(len(data_norm), 0, -1))

    try:
        autocorr_time = np.where(acf < 1/np.e)[0][0]
    except IndexError:
        autocorr_time = len(acf) // 10

    return autocorr_time


def plot_energy_convergence(results, equilibration=1000, args=None):
    """Plot energy convergence with error estimates"""
    energy_data = np.array(results["energy"])

    stats = calculate_energy_statistics(energy_data, equilibration)

    if stats is None:
        return

    plt.figure(figsize=(4, 3))

    plt.plot(energy_data[:, 0], energy_data[:, 1],
             'r-', alpha=0.5, label='Energy', linewidth=1)

    plt.axhline(y=stats['mean'], color='b', linestyle='-',
                label=f'Mean: {stats["mean"]:.6f}', linewidth=1)

    plt.axhline(y=stats['ci_95'][0], color='b', linestyle=':',
                label=f'95% CI: [{stats["ci_95"][0]:.6f}, {stats["ci_95"][1]:.6f}]', linewidth=1)
    plt.axhline(y=stats['ci_95'][1], color='b', linestyle=':', linewidth=1)

    plt.axhline(y=0.5, color='k', linestyle='--', label='Theoretical (0.5)', linewidth=1)

    plt.axvline(x=equilibration, color='g', linestyle='--',
                label=f'Equilibration ({equilibration} Steps)', linewidth=1)

    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)

    # textstr = '\n'.join((
    #     f'Mean Energy = {stats["mean"]:.6f}',
    #     f'Std Dev = {stats["std_dev"]:.6f}',
    #     f'SEM = {stats["sem"]:.6f}',
    #     f'Abs Error = {stats["abs_error"]:.6f}',
    #     f'Rel Error = {stats["rel_error"]:.2f}%',
    #     f'Autocorr Time = {stats["autocorr_time"]}',
    #     f'Effective Samples = {stats["effective_samples"]:.1f}'
    # ))

    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # plt.text(0.95, 0.07, textstr, transform=plt.gca().transAxes,
    #          fontsize=9, va='bottom', ha='right', bbox=props)

    if args.save_dir:
        plt.savefig(f'{args.save_dir}/energy_convergence.pgf')
    if args.plot:
        plt.show()
    plt.close()


def plot_energy_histogram(results, equilibration=1000, args=None):
    """Plot histogram of energy values after equilibration"""
    energy_data = np.array(results["energy"])

    equilibrated_data = energy_data[energy_data[:, 0] >= equilibration, 1]

    if len(equilibrated_data) == 0:
        print("Warning: No data points after discarded period")
        return

    stats = calculate_energy_statistics(energy_data, equilibration)

    plt.figure(figsize=(4, 3))

    n, bins, patches = plt.hist(equilibrated_data, bins=30, density=True,
                                alpha=0.7, color='skyblue')

    x = np.linspace(min(equilibrated_data), max(equilibrated_data), 100)
    plt.plot(x, statistics.norm.pdf(x, stats['mean'], stats['std_dev']),
             'r-', linewidth=1, label='Normal Distribution Fit')

    plt.axvline(x=stats['mean'], color='b', linestyle='-',
                label=f'Mean: {stats["mean"]:.6f}')
    plt.axvline(x=stats['ci_95'][0], color='b', linestyle=':')
    plt.axvline(x=stats['ci_95'][1], color='b', linestyle=':',
                label=f'95% CI: [{stats["ci_95"][0]:.6f}, {stats["ci_95"][1]:.6f}]')

    plt.axvline(x=0.5, color='k', linestyle='--', label='Theoretical (0.5)')

    plt.xlabel('Energy')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if args.save_dir:
        plt.savefig(f'{args.save_dir}/energy_histogram.pgf')
    if args.plot:
        plt.show()
    plt.close()


def plot_energy_error_analysis(results, equilibration=1000, args=None):
    """Plot error analysis for energy estimates"""
    energy_data = np.array(results["energy"])

    stats = calculate_energy_statistics(energy_data, equilibration)

    if stats is None:
        return

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 3))

    equilibrated_data = energy_data[energy_data[:, 0] >= equilibration, 1]
    x = energy_data[energy_data[:, 0] >= equilibration, 0]
    running_mean = np.cumsum(equilibrated_data) / \
        np.arange(1, len(equilibrated_data) + 1)

    ax1.plot(x, running_mean, 'b-', label='Running Average', linewidth=1)
    ax1.axhline(y=0.5, color='k', linestyle='--', label='Theoretical (0.5)')

    running_std = np.array([np.std(equilibrated_data[:i+1]) / np.sqrt(i+1)
                           for i in range(len(equilibrated_data))])

    ax1.fill_between(x, running_mean - 1.96 * running_std,
                     running_mean + 1.96 * running_std,
                     color='b', alpha=0.2, label='95% Confidence Band')

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Running Average Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # data_norm = equilibrated_data - np.mean(equilibrated_data)
    # acf = np.correlate(data_norm, data_norm, mode='full')
    # acf = acf[len(data_norm)-1:] / (np.var(data_norm)
    #                                 * np.arange(len(data_norm), 0, -1))

    # max_lag = min(len(acf), 100)
    # ax2.plot(range(max_lag), acf[:max_lag], 'r-')
    # ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    # ax2.axhline(y=1/np.e, color='k', linestyle='--',
    #             label=f'1/e (τ = {stats["autocorr_time"]})')

    # ax2.set_xlabel('Lag')
    # ax2.set_ylabel('Autocorrelation')
    # ax2.set_title('Energy Autocorrelation Function')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.save_dir:
        plt.savefig(f'{args.save_dir}/energy_error_analysis.pgf')
    if args.plot:
        plt.show()
    plt.close()


def plot_vref_and_energy(results, args):
    """Plot reference potential and energy"""
    energy_data = np.array(results["energy"])
    vref_data = np.array(results["vref"])

    plt.figure(figsize=(4, 3))
    plt.plot(energy_data[:500, 0], energy_data[:500, 1],
             'r-', label='Mean Potential', linewidth=1)
    plt.plot(vref_data[:500, 0], vref_data[:500, 1], 'b--', label='Vref', linewidth=1, alpha=0.5)

    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    if args.save_dir:
        plt.savefig(f'{args.save_dir}/energy_vref.pgf')
    if args.plot:
        plt.show()
    plt.close()


def plot_walker_population(results, args):
    """Plot walker population over time"""
    walker_data = np.array(results["walker_count"])

    plt.figure(figsize=(4, 3))
    plt.plot(walker_data[:, 0], walker_data[:, 1], linewidth=1)

    plt.xlabel('Steps')
    plt.ylabel('Number of Walkers')
    plt.grid(True)
    if args.save_dir:
        plt.savefig(f'{args.save_dir}/walker_population.pgf')
    if args.plot:
        plt.show()
    plt.close()


def print_energy_statistics(results, equilibration=1000):
    """Print detailed energy statistics"""
    energy_data = np.array(results["energy"])
    stats = calculate_energy_statistics(energy_data, equilibration)

    if stats is None:
        return

    print("\n" + "="*50)
    print("ENERGY STATISTICS")
    print("="*50)
    print(f"Number of samples: {stats['n_samples']}")
    print(f"Steps discarded: {equilibration}")
    print(f"Theoretical ground state energy: 0.500000")
    print("-"*50)
    print(f"Mean energy: {stats['mean']:.6f}")
    print(f"Standard deviation: {stats['std_dev']:.6f}")
    print(f"Standard error of the mean: {stats['sem']:.6f}")
    print(f"95% confidence interval: [{
          stats['ci_95'][0]:.6f}, {stats['ci_95'][1]:.6f}]")
    print("-"*50)
    print(f"Absolute error: {stats['abs_error']:.6f}")
    print(f"Relative error: {stats['rel_error']:.2f}%")
    print("-"*50)
    print(f"Autocorrelation time: {stats['autocorr_time']} Steps")
    print(f"Effective number of independent samples: {
          stats['effective_samples']:.1f}")
    print(f"Corrected standard error: {stats['corrected_sem']:.6f}")
    print(f"Corrected 95% confidence interval: [{
          stats['corrected_ci_95'][0]:.6f}, {stats['corrected_ci_95'][1]:.6f}]")
    print("="*50)

    if stats['corrected_ci_95'][0] <= 0.5 <= stats['corrected_ci_95'][1]:
        print("✅ Theoretical value (0.5) is within the 95% confidence interval")
    else:
        print("❌ Theoretical value (0.5) is outside the 95% confidence interval")
    print("="*50)


def main():
    args = parse_args()

    if args.plot and not args.save_dir:
        print("Warning: No directory specified for saving plots, plots will be displayed only.")

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

    print("Starting quantum Monte Carlo simulation...")
    time_start = time.time()
    results = qwp.run_qmc_simulation(
        n0=args.n0,          # Initial number of walkers
        max_steps=args.max_steps,      # Number of Monte Carlo steps
        h_x=args.h_x,         # Step length
        w0=args.w0,         # Initial distribution width
    )
    print(f"Simulation complete. Elapsed time: {time.time() - time_start:.2f}s")

    energy_data = np.array(results["energy"])

    print_energy_statistics(results, args.equilibration)

    if args.plot or args.save_dir:
        plot_vref_and_energy(results, args)
        plot_wave_function(results, args.save_steps, args)
        plot_energy_convergence(results, args.equilibration, args)
        plot_walker_population(results, args)
        plot_energy_histogram(results, args.equilibration, args)
        plot_energy_error_analysis(results, args.equilibration, args)

    final_energy = energy_data[-1, 1]
    print(f"Final ground state energy estimate: {final_energy:.6f}")
    print("Theoretical ground state energy: 0.500000")
