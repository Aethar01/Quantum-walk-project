import argparse
import quantum_walk_project.walkers as qwp
import matplotlib.pyplot as plt
import numpy as np
import math
qwp = qwp.walkersv2


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-h_x', '--h_x', type=float, default=0.25,
                        help='Step size in x direction (default=0.25)')
    parser.add_argument('-h_tau', '--h_tau', type=float, default=0.0625,
                        help='Step size in tau direction (default=0.0625)')
    parser.add_argument('-nw', '--num_walkers', type=int,
                        default=10_000, help='Number of walkers (default=10_000)')
    parser.add_argument('-ms', '--max_steps', type=int, default=80,
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

    survival_counts, e0_estimates, e0_estimates_no_ln, active_walkers, final_walkers, active_walkers_at_all_steps = qwp.run_walkv2(
        args.h_x, args.h_tau, args.num_walkers, args.max_steps, args.potential)
    survival_counts = np.array(survival_counts)
    e0_estimates = np.array(e0_estimates)
    e0_estimates_no_ln = np.array(e0_estimates_no_ln)
    active_walkers = np.array(active_walkers)
    final_walkers = np.array(final_walkers)

    optimal_tau = 2

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


        # plot ground state wave function
        walkers_below_2tau = []
        tau_2_step = 2 / args.h_tau
        for i in range(len(active_walkers_at_all_steps)):
            if i == tau_2_step:
                walkers_below_2tau.append(active_walkers_at_all_steps[i])

        walkers_below_2tau = np.array(walkers_below_2tau[-1])


        bins = np.arange(-3, 3, 2*args.h_x)
        # find probability of arriving at each bin
        prob_dividedby_2h_x = np.zeros(len(bins))
        for i, bin in enumerate(bins):
            values_in_bin = walkers_below_2tau[(walkers_below_2tau >= bin) & (walkers_below_2tau < bin + 2*args.h_x)]
            prob_dividedby_2h_x[i] = len(values_in_bin) / (len(walkers_below_2tau) * 2*args.h_x)

        sum = np.sum(prob_dividedby_2h_x * 2*args.h_x)
        if not math.isclose(sum, 1, abs_tol=1e-2):
            raise 'Sum of probabilities is not 1; sum is ' + str(sum)

        plt.bar(bins, prob_dividedby_2h_x, width=2*args.h_x, align='edge')
        # ground state wave function
        x = np.arange(-3, 3, 0.01)
        y = 1/((2*np.pi)**0.5) * np.exp(-1/2 * (x**2))
        plt.plot(x, y, color='red')
        plt.ylabel('$\\psi_0(x)$')
        plt.xlabel('x')
        plt.show()

        # plot ground state wave function squared
        walkers_below_4tau = []
        tau_4_step = 4 / args.h_tau
        for i in range(len(active_walkers_at_all_steps)):
            if i == tau_4_step:
                walkers_below_4tau.append(active_walkers_at_all_steps[i])

        walkers_below_4tau = np.array(walkers_below_4tau[-1])


        bins = np.arange(-3, 3, 2*args.h_x)
        # find probability of arriving at each bin
        prob_dividedby_2h_x = np.zeros(len(bins))
        for i, bin in enumerate(bins):
            values_in_bin = walkers_below_4tau[(walkers_below_4tau >= bin) & (walkers_below_4tau < bin + 2*args.h_x)]
            prob_dividedby_2h_x[i] = len(values_in_bin) / (len(walkers_below_4tau) * 2*args.h_x)

        sum = np.sum(prob_dividedby_2h_x * 2*args.h_x)
        if not math.isclose(sum, 1, abs_tol=1e-2):
            raise 'Sum of probabilities is not 1; sum is ' + str(sum)

        plt.bar(bins, prob_dividedby_2h_x, width=2*args.h_x, align='edge')
        # ground state wave function
        x = np.arange(-3, 3, 0.01)
        y = 1/((np.pi)**0.5) * np.exp(-1 * (x**2))
        plt.plot(x, y, color='red')
        plt.ylabel('$|\\psi_0(x)|^2$')
        plt.xlabel('x')
        plt.show()



    e0_2 = e0_estimates[steps_to_tau(np.arange(len(e0_estimates)), args.h_tau) == optimal_tau][0]
    e0_2_no_ln = e0_estimates_no_ln[steps_to_tau(np.arange(len(e0_estimates_no_ln)), args.h_tau) == optimal_tau][0]

    print(f'E0 estimate at t={optimal_tau}: {e0_2}')
    print(f'E0 estimate without ln at t={optimal_tau}: {e0_2_no_ln}')
    print(f'Theoretical value: {(np.pi**2)/8}')
    print(f'Residual at t={optimal_tau}: {e0_2_no_ln - np.pi**2/8}')

