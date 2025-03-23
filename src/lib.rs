use indicatif::ProgressBar;
use pyo3::exceptions::PyTypeError;
use pyo3::{prelude::*, wrap_pymodule};

mod walkv1;
mod walkv2;
mod walkv3;
mod walkv4;

/// Result of a quantum walk simulation
#[pyclass(get_all)]
struct Result {
    j: i32,
    j_sq: f64,
    lambda: f64,
    lambda_j_sq: f64,
    num_walkers: usize,
    max_steps: usize,
    residual: f64,
}

/// Run a single quantum walk simulation
///
/// Args:
///     j: Lattice size parameter
///     num_walkers: Number of walkers to simulate
///     max_steps: Maximum number of steps per walker
///     threads: Number of threads to use
///     seed: Random seed (default=0)
#[pyfunction]
#[pyo3(signature = (j, num_walkers, max_steps, threads, seed=0))]
fn run_walk(
    j: i32,
    num_walkers: usize,
    max_steps: usize,
    threads: usize,
    seed: u64,
) -> PyResult<Result> {
    walkv1::run(j, num_walkers, max_steps, threads, None, seed)
        .map(|(j_sq, lambda, lambda_j_sq, residual)| Result {
            j,
            j_sq,
            lambda,
            lambda_j_sq,
            num_walkers,
            max_steps,
            residual,
        })
        .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))
}

/// Run multiple quantum walk simulations with parameter ranges
///
/// Args:
///     j_lower: Minimum lattice size
///     j_step: Step size for lattice size
///     j_upper: Maximum lattice size
///     num_walkers_lower: Minimum number of walkers
///     num_walkers_step: Step size for number of walkers
///     num_walkers_upper: Maximum number of walkers
///     max_steps_lower: Minimum steps per walker
///     max_steps_step: Step size for steps per walker
///     max_steps_upper: Maximum steps per walker
///     threads: Number of threads
///     seed: Random seed (default=0)
#[pyfunction]
#[pyo3(signature = (j_lower, j_step, j_upper, num_walkers_lower, num_walkers_step, num_walkers_upper, max_steps_lower, max_steps_step, max_steps_upper, threads, seed=0, quiet=false))]
fn run_many_run_walk(
    j_lower: usize,
    j_step: usize,
    j_upper: usize,
    num_walkers_lower: usize,
    num_walkers_step: usize,
    num_walkers_upper: usize,
    max_steps_lower: usize,
    max_steps_step: usize,
    max_steps_upper: usize,
    threads: usize,
    seed: u64,
    quiet: bool,
) -> PyResult<Vec<Result>> {
    // Calculate total iterations for capacity and progress bar
    let j_range = (j_upper - j_lower) / j_step + 1;
    let num_walkers_range = (num_walkers_upper - num_walkers_lower) / num_walkers_step + 1;
    let max_steps_range = (max_steps_upper - max_steps_lower) / max_steps_step + 1;
    let total_iterations = j_range * num_walkers_range * max_steps_range;
    
    // Pre-allocate results vector with appropriate capacity
    let mut results = Vec::with_capacity(total_iterations);
    
    // Create progress bar if not quiet
    let bar = if quiet {
        ProgressBar::hidden()
    } else {
        ProgressBar::new(total_iterations as u64)
    };

    for j in (j_lower..=j_upper).step_by(j_step) {
        for num_walkers in (num_walkers_lower..=num_walkers_upper).step_by(num_walkers_step) {
            for max_steps in (max_steps_lower..=max_steps_upper).step_by(max_steps_step) {
                // Run the simulation with current parameters
                let result = walkv1::run(
                    j as i32,
                    num_walkers,
                    max_steps,
                    threads,
                    None,
                    seed,
                );
                
                // Handle the result
                match result {
                    Ok((j_sq, lambda, lambda_j_sq, residual)) => {
                        // Increment progress bar
                        bar.inc(1);
                        results.push(Result {
                            j: j as i32,
                            j_sq,
                            lambda,
                            lambda_j_sq,
                            num_walkers,
                            max_steps,
                            residual,
                        });
                    }
                    Err(e) => {
                        bar.finish_with_message("Error encountered");
                        return Err(PyTypeError::new_err(format!("{:?}", e)));
                    }
                }
            }
        }
    }
    
    bar.finish_with_message(format!("Completed {} simulations", results.len()));
    Ok(results)
}

/// Module containing v1 quantum walk simulations
#[pymodule]
fn walkersv1(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_walk, m)?)?;
    m.add_function(wrap_pyfunction!(run_many_run_walk, m)?)?;
    m.add_class::<Result>()?;
    Ok(())
}

/// Run a full Monte Carlo diffusion simulation (1D).
///
/// Returns a tuple containing:
/// 	`survival_counts`: Number of surviving walkers at each step.
/// 	`energy.values`: Ground state energy estimates.
/// 	`energy.uncertainties`: Uncertainties in energy estimates.
/// 	`active_walkers`: Final positions of surviving walkers (1D).
/// 	`final_locations`: Positions and steps at which walkers were terminated (1D).
/// 	`walker_history`: Positions of all active walkers at each step (1D).
#[pyfunction]
#[pyo3(signature = (h_x=0.25, h_tau=0.0625, num_walkers=10_000, max_steps=32, potential_number=1))]
fn run_walkv2(
    h_x: f64,
    h_tau: f64,
    num_walkers: usize,
    max_steps: usize,
    potential_number: usize,
) -> PyResult<(
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<(f64, usize)>,
    Vec<Vec<f64>>
)> {
    walkv2::run(
        Some(h_x), 
        Some(h_tau), 
        Some(num_walkers), 
        Some(max_steps), 
        Some(potential_number)
    )
    .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))
}

/// Module containing v2 quantum walk simulations
#[pymodule]
fn walkersv2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_walkv2, m)?)?;
    Ok(())
}

/// Run a full Monte Carlo diffusion simulation for multiple dimensions.
///
/// The additional parameter `dimensions` specifies the number of spatial dimensions.
/// If `dimensions` is set to 1, this function delegates to the 1D simulation for
/// backward compatibility.
///
/// Returns a tuple containing:
/// 	`survival_counts`: Number of surviving walkers at each step.
/// 	`energy.values`: Ground state energy estimates.
/// 	`energy.uncertainties`: Uncertainties in energy estimates.
/// 	`active_walkers`: Final positions of surviving walkers (each position is a Vec).
/// 	`final_locations`: Positions and steps at which walkers were terminated (each position is a Vec).
/// 	`walker_history`: Positions of all active walkers at each step (multi-dimensional).
#[pyfunction]
#[pyo3(signature = (h_x=0.25, h_tau=0.0625, num_walkers=10_000, max_steps=32, potential_number=1, dimensions=1))]
fn run_walkv3(
    h_x: f64,
    h_tau: f64,
    num_walkers: usize,
    max_steps: usize,
    potential_number: usize,
    dimensions: usize,
) -> PyResult<(
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<(Vec<f64>, usize)>,
    Vec<Vec<Vec<f64>>>,
)> {
    walkv3::run_multi(
        Some(h_x), 
        Some(h_tau), 
        Some(num_walkers), 
        Some(max_steps), 
        Some(potential_number),
        Some(dimensions)
    )
    .map_err(|e| PyTypeError::new_err(format!("{:?}", e)))
}

/// Module containing v3 quantum walk simulations
#[pymodule]
fn walkersv3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_walkv3, m)?)?;
    Ok(())
}

/// Module containing v4 quantum walk simulations
#[pymodule]
fn walkersv4(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(walkv4::run_qmc_simulation, m)?)?;
    Ok(())
}
/// Main quantum walk project module
#[pymodule]
#[pyo3(name = "walkers")]
fn quantum_walk_project(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(walkersv1))?;
    m.add_wrapped(wrap_pymodule!(walkersv2))?;
    m.add_wrapped(wrap_pymodule!(walkersv3))?;
    m.add_wrapped(wrap_pymodule!(walkersv4))?;
    Ok(())
}

