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
usage: walkv1 [-h] [-nu NUM_WALKERS_UPPER] [-nl NUM_WALKERS_LOWER]
              [-ns NUM_WALKERS_STEP] [-ju J_UPPER] [-jl J_LOWER]
              [-js J_STEP] [-mu MAX_STEPS_UPPER]
              [-ml MAX_STEPS_LOWER] [-ms MAX_STEPS_STEP] [-s SEED]
              [-t THREADS] [-o OUTPUT] [-r RUNS] [-a]
              [-ar ANALYSE_RUNS] [-p] [-pb] [-v]

Quantum Walker Simulation

options:
  -h, --help            show this help message and exit
  -s, --seed SEED       Seed for RNG. If 0, seed is random. NOTE:
                        With seed, process is not multithreaded
                        (default: 0)
  -t, --threads THREADS
                        Number of threads to use. If 0, use all
                        available threads (default: 0)
  -o, --output OUTPUT   Output file name (default: None)
  -r, --runs RUNS       Number of times to run the simulation
                        (default: 5)
  -a, --analyse         Narrow down parameters automatically
                        (default: False)
  -ar, --analyse_runs ANALYSE_RUNS
                        Number of runs to analyse (default: 100)
  -p, --plot            Plot the results (default: False)
  -pb, --progressbar    Display progress bar (default: False)
  -v, --version         show program's version number and exit

Walker Parameters:
  -nu, --num_walkers_upper NUM_WALKERS_UPPER
                        Maximum number of random walks (default:
                        100)
  -nl, --num_walkers_lower NUM_WALKERS_LOWER
                        Minimum number of random walks (default:
                        100)
  -ns, --num_walkers_step NUM_WALKERS_STEP
                        Step size for number of random walks
                        (default: 100)

Lattice Parameters:
  -ju, --j_upper J_UPPER
                        Maximum lattice size J (boundary at ±J)
                        (default: 10)
  -jl, --j_lower J_LOWER
                        Minimum lattice size J (boundary at ±J)
                        (default: 10)
  -js, --j_step J_STEP  Step size for lattice size (default: 2)

Step Parameters:
  -mu, --max_steps_upper MAX_STEPS_UPPER
                        Maximum number of steps per walk (default:
                        10000)
  -ml, --max_steps_lower MAX_STEPS_LOWER
                        Minimum number of steps per walk (default:
                        100)
  -ms, --max_steps_step MAX_STEPS_STEP
                        Step size for number of steps per walk
                        (default: 100)
```

For simulation 2:
```
usage: walkv2 [-h] [-h_x H_X] [-nw NUM_WALKERS] [-ms MAX_STEPS] [-p]
              [-mt MAX_TAU] [-pot POTENTIAL] [-a] [-v]

Quantum Walker Simulation (v2)

options:
  -h, --help            show this help message and exit
  -h_x, --h_x H_X       Step size in x direction (default: 0.25)
  -nw, --num_walkers NUM_WALKERS
                        Number of walkers (default: 10000)
  -ms, --max_steps MAX_STEPS
                        Maximum number of steps per walk 80
                        (default: 96)
  -p, --plot            Plot the results (default: False)
  -mt, --max_tau MAX_TAU
                        Maximum of tau (auto-calculates and
                        overrides max_steps) (default: None)
  -pot, --potential POTENTIAL
                        Potential type: 1=0.5x^2, 2=x (default: 1)
  -a, --auto_analyze    Automatically analyze optimal parameters
                        (default: False)
  -v, --version         show program's version number and exit
```

For simulation 3:
```
usage: walkv3 [-h] [-n0 N0] [-ms MAX_STEPS] [-h_x H_X] [-w0 W0]
              [-ss SAVE_STEPS [SAVE_STEPS ...]] [-p]
              [-e EQUILIBRATION]

options:
  -h, --help            show this help message and exit
  -n0, --n0 N0          Initial number of walkers (default: 50)
  -ms, --max_steps MAX_STEPS
                        Number of Monte Carlo steps (default: 50000)
  -h_x, --h_x H_X       Step length (default: 0.1)
  -w0, --w0 W0          Initial distribution width (default: 2.0)
  -ss, --save_steps SAVE_STEPS [SAVE_STEPS ...]
                        Steps at which to save histograms (default:
                        [500, 50000])
  -p, --plot            Plot results (default: False)
  -e, --equilibration EQUILIBRATION
                        Number of steps to discard (default: 1000)
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
