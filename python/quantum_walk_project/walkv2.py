import argparse
import quantum_walk_project.walkers as qwp
import matplotlib.pyplot as plt
import numpy as np
qwp = qwp.walkersv2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-h_x', '--h_x', type=float, default=0.25, help='Step size in x direction (default=0.25)')
    parser.add_argument('-h_tau', '--h_tau', type=float, default=0.0625, help='Step size in tau direction (default=0.0625)')
    parser.add_argument('-nw', '--num_walkers', type=int, default=10_000, help='Number of walkers (default=10_000)')
    parser.add_argument('-ms', '--max_steps', type=int, default=32, help='Maximum number of steps per walk (default=32)')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the results')
    parser.add_argument('-mt', '--max_tau', type=float, help='Maximum of tau (auto-calculates and overrides max_steps)')

    return parser.parse_args()


def steps_to_tau(steps, h_tau):
    return steps * h_tau


def main():
    args = parse_arguments()

    if args.max_tau:
        args.max_steps = int(args.max_tau / args.h_tau)

    survival_counts, e0_estimates = qwp.run_walkv2(args.h_x, args.h_tau, args.num_walkers, args.max_steps)
    survival_counts = np.array(survival_counts)
    e0_estimates = np.array(e0_estimates)

    if args.plot:
        fig, ax1 = plt.subplots()
        ax1.plot(steps_to_tau(np.arange(len(survival_counts)), args.h_tau), survival_counts/args.num_walkers, label='Survival probabilty', color='tab:blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Survival probabilty', color='tab:blue')

        ax2 = ax1.twinx()
        ax2.scatter(steps_to_tau(np.arange(len(e0_estimates)), args.h_tau), e0_estimates, label='$E_0$ estimate', color='tab:red', marker='x', s=10)
        ax2.set_ylabel('$E_0$ estimate', color='tab:red')

        plt.show()

    print(f'Final survival count: {survival_counts[-1]}')
    print(f'Final E0 estimate: {e0_estimates[-1]}')
    print('Cutting initial E0 estimates:')

    e0_estimates = e0_estimates[e0_estimates >= 0.4]

    print(f'Average E0 estimate: {np.mean(e0_estimates)}')
