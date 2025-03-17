import argparse
import quantum_walk_project.walkers as qwp
import matplotlib.pyplot as plt
import numpy as np
qwp = qwp.walkersv2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-h_x', '--h_x', type=float, default=0.25,
                        help='Step size in x direction (default=0.25)')
    parser.add_argument('-h_tau', '--h_tau', type=float, default=0.0625,
                        help='Step size in tau direction (default=0.0625)')
    parser.add_argument('-nw', '--num_walkers', type=int,
                        default=10_000, help='Number of walkers (default=10_000)')
    parser.add_argument('-ms', '--max_steps', type=int, default=32,
                        help='Maximum number of steps per walk (default=32)')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the results')
    parser.add_argument('-mt', '--max_tau', type=float,
                        help='Maximum of tau (auto-calculates and overrides max_steps)')
    parser.add_argument('-pot', '--potential', type=int, default=1,
                        help='Potential type: 1=0.5x^2, 2=x, (default=1)')

    return parser.parse_args()


def steps_to_tau(steps, h_tau):
    return steps * h_tau


def main():
    args = parse_arguments()

    if args.max_tau:
        args.max_steps = int(args.max_tau / args.h_tau)

    survival_counts, e0_estimates, e0_estimates_no_ln = qwp.run_walkv2(
        args.h_x, args.h_tau, args.num_walkers, args.max_steps, args.potential)
    survival_counts = np.array(survival_counts)
    e0_estimates = np.array(e0_estimates)
    e0_estimates_no_ln = np.array(e0_estimates_no_ln)

    if args.plot:
        fig, ax1 = plt.subplots()
        ax1.plot(steps_to_tau(np.arange(len(survival_counts)), args.h_tau),
                 survival_counts/args.num_walkers, label='Survival probabilty', color='tab:blue')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Survival probabilty', color='tab:blue')

        ax2 = ax1.twinx()
        ax2.scatter(steps_to_tau(np.arange(len(e0_estimates)), args.h_tau),
                    e0_estimates, label='$E_0$ estimate', color='tab:red', marker='x', s=10)
        ax2.set_ylabel('$E_0$ estimate', color='tab:red')

        plt.show()

        # plot residuals
        fig, ax = plt.subplots()
        ax.plot(steps_to_tau(np.arange(len(e0_estimates_no_ln)),
                args.h_tau), e0_estimates_no_ln - np.pi**2/8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual')
        plt.show()

    print(f'Average E0 estimate: {np.mean(e0_estimates)}')
    print(f'Average E0 estimate without ln: {np.mean(e0_estimates_no_ln)}')

    # find when dE0/E0 < 0.01
    for i in range(1, len(e0_estimates)):
        if abs((e0_estimates[i] - e0_estimates[i-1])/e0_estimates[i]) < 0.001:
            optimal_tau = steps_to_tau(i, args.h_tau)
            print(f'Optimum time to measure E0 is t={steps_to_tau(i, args.h_tau)}')
            break

    # print('Optimum time to measure E0 is t=2')
    e0_2 = e0_estimates[steps_to_tau(np.arange(len(e0_estimates)), args.h_tau) == optimal_tau][0]
    e0_2_no_ln = e0_estimates_no_ln[steps_to_tau(np.arange(len(e0_estimates_no_ln)), args.h_tau) == optimal_tau][0]

    print(f'E0 estimate at t={optimal_tau}: {e0_2}')
    print(f'E0 estimate without ln at t={optimal_tau}: {e0_2_no_ln}')
    print(f'Theoretical value: {(np.pi**2)/8}')
    print(f'Residual at t={optimal_tau}: {e0_2_no_ln - np.pi**2/8}')

