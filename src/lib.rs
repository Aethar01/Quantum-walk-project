use indicatif::ProgressBar;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

mod walk;

/// Result { j_sq, lambda_j_sq, num_walkers, max_steps, residual }
#[pyclass{get_all}]
struct Result {
    j: i32,
    j_sq: f64,
    lambda_j_sq: f64,
    num_walkers: usize,
    max_steps: usize,
    residual: f64,
}

/// Inputs: j, num_walkers, max_steps, threads, seed
/// Returns: Result { j_sq, lambda_j_sq, num_walker, max_steps, residual }
#[pyfunction]
#[pyo3(signature = (j, num_walkers, max_steps, threads, seed=0))]
fn run_walk(
    j: i32,
    num_walkers: usize,
    max_steps: usize,
    threads: usize,
    seed: u64,
) -> PyResult<Result> {
    match walk::run(j, num_walkers, max_steps, threads, None, seed) {
        Ok((j_sq, lambda_j_sq, residual)) => Ok(Result {
            j,
            j_sq,
            lambda_j_sq,
            num_walkers,
            max_steps,
            residual,
        }),
        Err(e) => Err(PyTypeError::new_err(format!("{:?}", e))),
    }
}

/// Inputs: j_lower, j_step, j_upper, num_walkers_lower, num_walkers_step, num_walkers_upper,
/// max_steps_lower, max_steps_step, max_steps_upper, threads, seed
#[pyfunction]
#[pyo3(signature = (j_lower, j_step, j_upper, num_walkers_lower, num_walkers_step, num_walkers_upper, max_steps_lower, max_steps_step, max_steps_upper, threads, seed=0))]
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
) -> PyResult<Vec<Result>> {
    let mut results = Vec::new();
    let bar = ProgressBar::new(
        (((j_upper - j_lower) / j_step
            + 1) * ((num_walkers_upper - num_walkers_lower) / num_walkers_step
            + 1 ) * ((max_steps_upper - max_steps_lower) / max_steps_step
            + 1)) as u64,
    );
    for j in (j_lower..=j_upper).step_by(j_step as usize) {
        for num_walkers in (num_walkers_lower..=num_walkers_upper).step_by(num_walkers_step as usize) {
            for max_steps in (max_steps_lower..=max_steps_upper).step_by(max_steps_step as usize) {
                match walk::run(
                    j as i32,
                    num_walkers as usize,
                    max_steps as usize,
                    threads as usize,
                    None,
                    seed,
                ) {
                    Ok((j_sq, lambda_j_sq, residual)) => {
                        bar.inc(1);
                        results.push(Result {
                            j: j as i32,
                            j_sq,
                            lambda_j_sq,
                            num_walkers: num_walkers as usize,
                            max_steps: max_steps as usize,
                            residual,
                        })
                    }
                    Err(e) => {
                        bar.inc(1);
                        return Err(PyTypeError::new_err(format!("{:?}", e)));
                    }
                }
            }
        }
    }
    Ok(results)
}

#[pymodule]
fn quantum_walk_project(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_walk, m)?)?;
    m.add_function(wrap_pyfunction!(run_many_run_walk, m)?)?;
    m.add_class::<Result>()?;
    Ok(())
}
