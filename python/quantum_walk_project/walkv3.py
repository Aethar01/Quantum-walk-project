import argparse
import matplotlib.pyplot as plt
import numpy as np
import quantum_walk_project.walkers as qwp
qwp = qwp.walkersv3


def parse_args():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n0', '--n0', type=int, default=50,
                        help='Initial number of walkers')
    parser.add_argument('-mcs', '--mcs', type=int, default=50000,
                        help='Number of Monte Carlo steps')
    parser.add_argument('-ds', '--ds', type=float,
                        default=0.1, help='Step length')
    parser.add_argument('-w0', '--w0', type=float, default=2.0,
                        help='Initial distribution width')
    parser.add_argument('-ss', '--save_steps', type=int, nargs='+',
                        default=[500, 50000], help='Steps at which to save histograms')
    parser.add_argument('-dt', '--dt_factor', type=float,
                        default=1, help='Time step factor')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot results')
    return parser.parse_args()


def calculate_histogram(walker_positions, xmin=-6.0, xmax=6.0, nbins=100):
    """Calculate histogram from walker positions"""
    hist, bin_edges = np.histogram(
        walker_positions, bins=nbins, range=(xmin, xmax), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist


def collect_walkers_before_step(results, target_step):
    """Collect walker positions from steps up to the target step"""

    in_range_steps = [s for s in range(1, target_step+1)]

    # Collect all walker positions
    all_walkers = []
    for step in in_range_steps:
        walkers = np.array(results[f'walkers_{step}'])
        all_walkers.extend(walkers.flatten())

    return np.array(all_walkers), step


def plot_wave_function(results, target_steps):
    """Plot wave function at specific steps"""
    # Generate exact solution
    xmin, xmax = -6.0, 6.0
    x_exact = np.linspace(xmin, xmax, 200)
    psi_exact = np.pi**(-0.25) * np.exp(-0.5 * x_exact**2)

    plt.figure(figsize=(10, 6))

    # Plot exact solution
    plt.plot(x_exact, psi_exact, 'k-', label='Exact')

    # Plot histograms for each target step
    colors = ['b-', 'g--', 'r-.', 'm:']
    for i, step in enumerate(target_steps):
        # Collect walker positions around target step
        walkers, step = collect_walkers_before_step(results, step)

        # Calculate histogram
        if len(walkers) > 0:
            x_bins, hist = calculate_histogram(walkers)
            plt.plot(x_bins, hist, colors[i %
                     len(colors)], label=f'Step {step}')
            # plt.bar(x_bins, hist, width=0.1, alpha=0.5, label=f'Step {step}')

    plt.xlabel('x')
    plt.ylabel('psi(x)')
    plt.title('Quantum Monte Carlo Wave Function')
    plt.legend()
    plt.grid(True)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.8)
    plt.show()
    plt.close()


def plot_energy_convergence(results):
    """Plot energy convergence"""
    energy_data = np.array(results["energy"])

    plt.figure(figsize=(10, 6))
    plt.plot(energy_data[:, 0], energy_data[:, 1], 'r-', label='Energy')
    plt.axhline(y=0.5, color='k', linestyle='--', label='Exact (0.5)')

    plt.xlabel('MC steps')
    plt.ylabel('Energy')
    plt.title('Quantum Monte Carlo Energy Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def plot_vref_and_energy(results):
    """Plot reference potential and energy"""
    energy_data = np.array(results["energy"])
    vref_data = np.array(results["vref"])

    plt.figure(figsize=(10, 6))
    plt.plot(energy_data[:500, 0], energy_data[:500, 1],
             'r-', label='Esum/imcs')
    plt.plot(vref_data[:500, 0], vref_data[:500, 1], 'b--', label='Vref')

    plt.xlabel('MC steps')
    plt.ylabel('Energy')
    plt.title('Quantum Monte Carlo Energy and Vref')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()


def plot_walker_population(results):
    """Plot walker population over time"""
    walker_data = np.array(results["walker_count"])

    plt.figure(figsize=(10, 6))
    plt.plot(walker_data[:, 0], walker_data[:, 1])

    plt.xlabel('MC steps')
    plt.ylabel('Number of Walkers')
    plt.title('Walker Population Over Time')
    plt.grid(True)
    plt.show()
    plt.close()


def plot_results(energy_data, vref_data, walker_count, histograms, exact_solution, save_steps):
    # Plot energy and Vref (first 500 steps)
    plt.figure(figsize=(10, 6))
    plt.plot(energy_data[:500, 0], energy_data[:500, 1],
             'r-', label='Esum/imcs')
    plt.plot(vref_data[:500, 0], vref_data[:500, 1], 'b--', label='Vref')
    plt.xlabel('MC steps')
    plt.ylabel('Energy')
    plt.title('Quantum Monte Carlo Energy Convergence')
    plt.legend()
    plt.grid(True)
    plt.ylim(-6, 6)
    plt.show()
    plt.close()

    # Plot walker distribution compared to exact solution

    plt.figure(figsize=(10, 6))
    # Plot histograms for each save step
    colors = ['b-', 'g--', 'm-.', 'c:']
    for i, step in enumerate(save_steps):
        plt.plot(histograms[step][:, 0], histograms[step][:, 1],
                 colors[i % len(colors)], label=f'mcs={step}')
    print(histograms)
    plt.plot(exact_solution[:, 0], exact_solution[:, 1], 'r-', label='exact')
    plt.xlabel('x')
    plt.ylabel('psi(x)')
    plt.title('Quantum Monte Carlo Walker Distribution')
    plt.legend()
    plt.grid(True)
    plt.xlim(-6, 6)
    plt.ylim(0, 0.8)
    plt.show()
    plt.close()

    # Plot number of walkers over time
    plt.figure(figsize=(10, 6))
    plt.plot(walker_count[:, 0], walker_count[:, 1])
    plt.xlabel('MC steps')
    plt.ylabel('Number of walkers')
    plt.title('Walker Population Over Time')
    plt.grid(True)
    plt.show()
    plt.close()


def main():
    args = parse_args()

    # Run the quantum Monte Carlo simulation
    print("Starting quantum Monte Carlo simulation...")
    results = qwp.run_qmc_simulation(
        n0=args.n0,          # Initial number of walkers
        mcs=args.mcs,      # Number of Monte Carlo steps
        ds=args.ds,         # Step length
        w0=args.w0,         # Initial distribution width
        dt_factor=args.dt_factor,  # Time step factor
    )

    # Extract data
    energy_data = np.array(results["energy"])
    # vref_data = np.array(results["vref"])
    # walker_count = np.array(results["walker_count"])
    # exact_solution = np.array(results["exact_solution"])

    if args.plot:
        plot_vref_and_energy(results)
        plot_wave_function(results, args.save_steps)
        plot_energy_convergence(results)
        plot_walker_population(results)

    # Extract histograms for each save step
    # histograms = {}
    # for step in args.save_steps:
    #     histograms[step] = np.array(results[f"histogram_{step}"])

    # if args.plot:
    #     plot_results(energy_data, vref_data, walker_count,
    #                  histograms, exact_solution, args.save_steps)

    # Calculate final ground state energy estimate
    final_energy = energy_data[-1, 1]
    print(f"Final ground state energy estimate: {final_energy:.6f}")
    print("Theoretical ground state energy: 0.500000")
