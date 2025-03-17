# Quantum-walk-project

This project aims to simulate the ground state of a quantum system within a 1D lattice using random walk. 

## Build

Dependencies:
- [python3](https://www.python.org/)
- [python3-venv](https://docs.python.org/3/library/venv.html)

Optional dependencies:
- [just](https://github.com/casey/just)

To build the project, run either of the following commands:

```bash
just build
```

or

```bash
./run -b
```

## Run

To run the project, execute the following command:

```bash
just run{int} [options]
```
or
```bash
./run -r {int} -- [options]
```

The options are:

For simulation 1:
```
usage: walkv1 [-h] [-nu NUM_WALKERS_UPPER] [-nl NUM_WALKERS_LOWER] [-ns NUM_WALKERS_STEP] [-ju J_UPPER] [-jl J_LOWER] [-js J_STEP] [-mu MAX_STEPS_UPPER]
                            [-ml MAX_STEPS_LOWER] [-ms MAX_STEPS_STEP] [-s SEED] [-t THREADS] [-o OUTPUT] [-r RUNS] [-a]

options:
  -h, --help            show this help message and exit
  -nu, --num_walkers_upper NUM_WALKERS_UPPER
                        Maximum number of random walks to simulate (Inclusive, default=1000)
  -nl, --num_walkers_lower NUM_WALKERS_LOWER
                        Minimum number of random walks to simulate (Inclusive, default=100)
  -ns, --num_walkers_step NUM_WALKERS_STEP
                        Step size for number of random walks (default=100)
  -ju, --j_upper J_UPPER
                        Maximum lattice size J (boundary at ±J) (Inclusive, default=20)
  -jl, --j_lower J_LOWER
                        Minimum lattice size J (boundary at ±J) (Inclusive, default=6)
  -js, --j_step J_STEP  Step size for lattice size (default=2)
  -mu, --max_steps_upper MAX_STEPS_UPPER
                        Maximum number of steps per walk (Inclusive, default=1000)
  -ml, --max_steps_lower MAX_STEPS_LOWER
                        Minimum number of steps per walk (Inclusive, default=100)
  -ms, --max_steps_step MAX_STEPS_STEP
                        Step size for number of steps per walk (default=100)
  -s, --seed SEED       Seed for random number generator. If 0, seed is random. NOTE: IF A SEED IS PROVIDED, THE PROCESS IS NOT MULTITHREADED (default=0)
  -t, --threads THREADS
                        Number of threads to use. If 0, use all available threads (default=0)
  -o, --output OUTPUT   Output file name
  -r, --runs RUNS       Number of times to run the simulation (default=1)
  -a, --analyse         Narrow down paramenters automatically
```

For simulation 2:
```
usage: walkv2 [-h] [-h_x H_X] [-h_tau H_TAU] [-nw NUM_WALKERS]
              [-ms MAX_STEPS] [-p] [-mt MAX_TAU] [-pot POTENTIAL]

options:
  -h, --help            show this help message and exit
  -h_x, --h_x H_X       Step size in x direction (default=0.25)
  -h_tau, --h_tau H_TAU
                        Step size in tau direction (default=0.0625)
  -nw, --num_walkers NUM_WALKERS
                        Number of walkers (default=10_000)
  -ms, --max_steps MAX_STEPS
                        Maximum number of steps per walk
                        (default=32)
  -p, --plot            Plot the results
  -mt, --max_tau MAX_TAU
                        Maximum of tau (auto-calculates and
                        overrides max_steps)
  -pot, --potential POTENTIAL
                        Potential type: 1=0.5x^2, 2=x, (default=1)
```

## Clean

To clean the project, deleting the venv and target build directories, run either of the following commands:

```bash
just clean
```
or
```bash
./run -c
```
