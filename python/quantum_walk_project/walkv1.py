import argparse
import quantum_walk_project.walkers as qwp
import pandas as pd
import matplotlib.pyplot as plt
import math
qwp = qwp.walkersv1


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-nu', '--num_walkers_upper', type=int,
                        default=1000, help='Maximum number of random walks to simulate (Inclusive, default=1000)')
    parser.add_argument('-nl', '--num_walkers_lower', type=int,
                        default=100, help='Minimum number of random walks to simulate (Inclusive, default=100)')
    parser.add_argument('-ns', '--num_walkers_step', type=int,
                        default=100, help='Step size for number of random walks (default=100)')

    parser.add_argument('-ju', '--j_upper', type=int, default=20,
                        help='Maximum lattice size J (boundary at ±J) (Inclusive, default=20)')
    parser.add_argument('-jl', '--j_lower', type=int, default=6,
                        help='Minimum lattice size J (boundary at ±J) (Inclusive, default=6)')
    parser.add_argument('-js', '--j_step', type=int,
                        default=2, help='Step size for lattice size (default=2)')

    parser.add_argument('-mu', '--max_steps_upper', type=int,
                        default=1000, help='Maximum number of steps per walk (Inclusive, default=1000)')
    parser.add_argument('-ml', '--max_steps_lower', type=int,
                        default=100, help='Minimum number of steps per walk (Inclusive, default=100)')
    parser.add_argument('-ms', '--max_steps_step', type=int,
                        default=100,  help='Step size for number of steps per walk (default=100)')

    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Seed for random number generator. If 0, seed is random. NOTE: IF A SEED IS PROVIDED, THE PROCESS IS NOT MULTITHREADED (default=0)')
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help='Number of threads to use. If 0, use all available threads (default=0)')

    parser.add_argument('-o', '--output', type=str, help='Output file name')

    parser.add_argument('-r', '--runs', type=int, default=1,
                        help='Number of times to run the simulation (default=1)')

    parser.add_argument('-a', '--analyse', action='store_true',
                        help='Narrow down paramenters automatically')

    return parser.parse_args()


def init_data_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=['j', 'num_walkers', 'max_steps', 'j_sq', 'lambda_j_sq', 'residual'])


def append_data_frame(df: pd.DataFrame, result: qwp.Result) -> pd.DataFrame:
    return pd.concat([df if not df.empty else None, pd.DataFrame([{
        'j': result.j,
        'num_walkers': result.num_walkers,
        'max_steps': result.max_steps,
        'j_sq': result.j_sq,
        'lambda_j_sq': result.lambda_j_sq,
        'residual': result.residual
    }])], ignore_index=True)


def auto_analyse(args, df, theoretical):
    # find the best parameters
    min_residual = df['residual'].min()
    best = df[df['residual'] == min_residual]
    print(f'Best Parameters: \n{best}')
    return best


def plot_results(df: pd.DataFrame, theoretical: float):
    # plot to see what values yield the lowest residuals
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].scatter(df['j'], df['residual'])
    ax[0, 0].set_xlabel('J')
    ax[0, 0].set_ylabel('Residual')
    ax[0, 0].grid()

    ax[0, 1].scatter(df['num_walkers'], df['residual'])
    ax[0, 1].set_xlabel('Number of Walkers')
    ax[0, 1].set_ylabel('Residual')
    ax[0, 1].grid()

    ax[1, 0].scatter(df['max_steps'], df['residual'])
    ax[1, 0].set_xlabel('Max Steps')
    ax[1, 0].set_ylabel('Residual')
    ax[1, 0].grid()

    ax[1, 1].scatter(df['j_sq'], df['residual'])
    ax[1, 1].set_xlabel('$J^2$')
    ax[1, 1].set_ylabel('Residual')
    ax[1, 1].grid()

    plt.show()

    # histogram lambda_j_sq and theoretical value
    fig, ax = plt.subplots()
    ax.hist(df['lambda_j_sq'], bins=50, alpha=0.5, label='Simulated')
    ax.axvline(x=theoretical, color='r', label='Theoretical: $π^2/8$')
    ax.set_xlabel('$λJ^2$')
    ax.set_ylabel('Frequency')
    ax.grid()
    ax.legend()
    plt.show()


def generate_data_frame(args):
    df = init_data_frame()
    for _ in range(args.runs):
        result = qwp.run_many_run_walk(
            args.j_lower, args.j_step, args.j_upper,
            args.num_walkers_lower, args.num_walkers_step, args.num_walkers_upper,
            args.max_steps_lower, args.max_steps_step, args.max_steps_upper,
            threads=args.threads, seed=args.seed
        )
        for r in result:
            df = append_data_frame(df, r)
    return df


def main():
    args = parse_arguments()

    theoretical = (math.pi ** 2) / 8

    df = generate_data_frame(args)

    if args.analyse:
        af = auto_analyse(args, df, theoretical)
        args.j_lower = af['j'].values[0]
        args.j_upper = af['j'].values[0]
        args.num_walkers_lower = af['num_walkers'].values[0]
        args.num_walkers_upper = af['num_walkers'].values[0]
        args.max_steps_lower = af['max_steps'].values[0]
        args.max_steps_upper = af['max_steps'].values[0]
        args.runs = 100
        df = generate_data_frame(args)

    if args.output:
        df.to_csv(f'{args.output}', index=False)

    plot_results(df, theoretical)

    average_lambda_j_sq = df['lambda_j_sq'].mean()
    print('Theoretical Value (π^2/8):', round(theoretical, 4))
    print('Average λJ^2:', round(average_lambda_j_sq, 4))
    print('Residual of Average λJ^2:', round(abs(
        theoretical - average_lambda_j_sq), 4))
