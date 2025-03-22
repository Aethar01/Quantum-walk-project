use anyhow::Result;
use rand::prelude::*;

/// Harmonic oscillator potential: 0.5 * x² (for 1D)
fn potential_harmonic(x: f64) -> f64 {
    0.5 * x * x
}

/// Alternative potential (originally documented as "x²" but implemented as x in the
/// sample); we keep it as is for 1D.
fn potential_linear(x: f64) -> f64 {
    x
}

/// Multi-dimensional harmonic oscillator potential:
/// \( V(\mathbf{x}) = \frac{1}{2}\sum_{i=1}^{d} x_i^2 \)
fn potential_harmonic_nd(x: &[f64]) -> f64 {
    0.5 * x.iter().map(|xi| xi * xi).sum::<f64>()
}

/// Multi-dimensional alternative potential. Here we simply sum the coordinates,
/// following the original 1D version. (In practice you might choose a different
/// definition for a “linear” potential in several dimensions.)
fn potential_linear_nd(x: &[f64]) -> f64 {
    x.iter().sum()
}

/// Represents simulation results from diffusion Monte Carlo (1D version)
struct SimulationResult {
    /// Number of surviving walkers at each step
    survival_counts: Vec<usize>,
    /// Final positions of surviving walkers (1D)
    active_walkers: Vec<f64>,
    /// Positions and steps at which walkers were terminated (1D)
    final_locations: Vec<(f64, usize)>,
    /// Positions of all active walkers at each step (1D)
    walker_history: Vec<Vec<f64>>,
}

/// Represents simulation results for multi-dimensional simulations
struct SimulationResultMulti {
    /// Number of surviving walkers at each step
    survival_counts: Vec<usize>,
    /// Final positions of surviving walkers (each walker is a vector of f64)
    active_walkers: Vec<Vec<f64>>,
    /// Positions and steps at which walkers were terminated (each position is a Vec)
    final_locations: Vec<(Vec<f64>, usize)>,
    /// Positions of all active walkers at each step (multi-dimensional)
    walker_history: Vec<Vec<Vec<f64>>>,
}

/// Represents ground state energy estimates
struct EnergyEstimate {
    /// Energy estimates
    values: Vec<f64>,
    /// Uncertainties in energy estimates
    values_no_ln: Vec<f64>,
}

/// Run diffusion Monte Carlo simulation with walkers in 1D (unchanged for
/// backward compatibility).
fn run_walkers(
    n_walkers: usize,
    max_steps: usize,
    h_x: f64,
    h_tau: f64,
    potential_number: usize,
) -> SimulationResult {
    let potential: fn(f64) -> f64 = match potential_number {
        1 => potential_harmonic,
        2 => potential_linear,
        _ => panic!("Invalid potential function"),
    };

    let mut rng = rand::rng();
    let mut active_walkers = vec![0.0; n_walkers];
    let mut survival_counts = Vec::with_capacity(max_steps);
    let mut final_locations: Vec<(f64, usize)> = Vec::with_capacity(n_walkers);
    let mut walker_history: Vec<Vec<f64>> = Vec::new();

    for step in 0..max_steps {
        let mut next_walkers = Vec::with_capacity(active_walkers.len());

        for &x in &active_walkers {
            let x_new = x + if rng.random_bool(0.5) { h_x } else { -h_x };
            let a = h_tau * potential(x_new);

            if rng.random::<f64>() >= a {
                next_walkers.push(x_new);
            } else {
                final_locations.push((x_new, step));
            }
        }

        survival_counts.push(next_walkers.len());
        walker_history.push(next_walkers.clone());
        active_walkers = next_walkers;
    }

    SimulationResult {
        survival_counts,
        active_walkers,
        final_locations,
        walker_history,
    }
}

/// Run diffusion Monte Carlo simulation with walkers in multiple dimensions.
/// In each simulation step every coordinate is updated independently (random
/// ±hₓ shift).
fn run_walkers_multi(
    n_walkers: usize,
    max_steps: usize,
    h_x: f64,
    h_tau: f64,
    potential_number: usize,
    n_dim: usize,
) -> SimulationResultMulti {
    let potential: fn(&[f64]) -> f64 = match potential_number {
        1 => potential_harmonic_nd,
        2 => potential_linear_nd,
        _ => panic!("Invalid potential function"),
    };

    let mut rng = rand::rng();
    // Initialize each walker at the origin in n_dim dimensions.
    let mut active_walkers = vec![vec![0.0; n_dim]; n_walkers];
    let mut survival_counts = Vec::with_capacity(max_steps);
    let mut final_locations: Vec<(Vec<f64>, usize)> = Vec::with_capacity(n_walkers);
    let mut walker_history: Vec<Vec<Vec<f64>>> = Vec::new();

    for step in 0..max_steps {
        let mut next_walkers = Vec::with_capacity(active_walkers.len());

        for walker in &active_walkers {
            let mut new_position = walker.clone();
            // Update every coordinate independently.
            for coord in new_position.iter_mut() {
                *coord += if rng.random_bool(0.5) { h_x } else { -h_x };
            }
            let a = h_tau * potential(&new_position);

            if rng.random::<f64>() >= a {
                next_walkers.push(new_position);
            } else {
                final_locations.push((new_position, step));
            }
        }

        survival_counts.push(next_walkers.len());
        walker_history.push(next_walkers.clone());
        active_walkers = next_walkers;
    }

    SimulationResultMulti {
        survival_counts,
        active_walkers,
        final_locations,
        walker_history,
    }
}

/// Calculate ground state energy estimates based solely on the survival counts.
fn estimate_ground_state_energy(
    survival_counts: &[usize],
    delta_tau_steps: usize,
    h_tau: f64,
) -> EnergyEstimate {
    let mut values = Vec::new();
    let mut values_no_ln = Vec::new();

    for i in 0..(survival_counts.len() - delta_tau_steps) {
        let current = survival_counts[i] as f64;
        let future = survival_counts[i + delta_tau_steps] as f64;

        if current > 0.0 && future > 0.0 {
            let ratio = future / current;
            values.push(-ratio.ln() / (delta_tau_steps as f64 * h_tau));
            values_no_ln.push(ratio / (delta_tau_steps as f64 * h_tau));
        } else {
            values.push(0.0);
            values_no_ln.push(0.0);
        }
    }

    EnergyEstimate {
        values,
        values_no_ln,
    }
}

/// Run a full Monte Carlo diffusion simulation (1D).
///
/// Returns a tuple containing:
/// - `survival_counts`: Number of surviving walkers at each step.
/// - `energy.values`: Ground state energy estimates.
/// - `energy.values_no_ln`: Uncertainties in energy estimates.
/// - `active_walkers`: Final positions of surviving walkers (1D).
/// - `final_locations`: Positions and steps at which walkers were terminated (1D).
/// - `walker_history`: Positions of all active walkers at each step (1D).
pub fn run(
    h_x: Option<f64>,
    h_tau: Option<f64>,
    num_walkers: Option<usize>,
    max_steps: Option<usize>,
    potential_number: Option<usize>,
) -> Result<(
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<(f64, usize)>,
    Vec<Vec<f64>>,
)> {
    // Simulation parameters with defaults
    let h_x = h_x.unwrap_or(0.25);
    let h_tau = h_tau.unwrap_or(h_x * h_x);
    let n_walkers = num_walkers.unwrap_or(10_000);
    let max_steps = max_steps.unwrap_or(32);
    let delta_tau_steps = 10; // Δτ = 10 * h_tau
    let potential_number = potential_number.unwrap_or(1);

    let result = run_walkers(n_walkers, max_steps, h_x, h_tau, potential_number);
    let energy = estimate_ground_state_energy(&result.survival_counts, delta_tau_steps, h_tau);

    Ok((
        result.survival_counts,
        energy.values,
        energy.values_no_ln,
        result.active_walkers,
        result.final_locations,
        result.walker_history,
    ))
}

/// Run a full Monte Carlo diffusion simulation for multiple dimensions.
///
/// The additional parameter `dimensions` specifies the number of spatial dimensions.
/// If `dimensions` is set to 1, this function delegates to the 1D simulation for
/// backward compatibility.
///
/// Returns a tuple containing:
/// - `survival_counts`: Number of surviving walkers at each step.
/// - `energy.values`: Ground state energy estimates.
/// - `energy.values_no_ln`: Uncertainties in energy estimates.
/// - `active_walkers`: Final positions of surviving walkers (each position is a Vec).
/// - `final_locations`: Positions and steps at which walkers were terminated
///   (each position is a Vec).
/// - `walker_history`: Positions of all active walkers at each step (multi-dimensional).
pub fn run_multi(
    h_x: Option<f64>,
    h_tau: Option<f64>,
    num_walkers: Option<usize>,
    max_steps: Option<usize>,
    potential_number: Option<usize>,
    dimensions: Option<usize>,
) -> Result<(
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<(Vec<f64>, usize)>,
    Vec<Vec<Vec<f64>>>,
)> {
    let h_x = h_x.unwrap_or(0.25);
    let h_tau = h_tau.unwrap_or(h_x * h_x);
    let n_walkers = num_walkers.unwrap_or(10_000);
    let max_steps = max_steps.unwrap_or(32);
    let n_dim = dimensions.unwrap_or(1);
    let delta_tau_steps = 10; // Δτ = 10 * h_tau
    let potential_number = potential_number.unwrap_or(1);

    if n_dim == 1 {
        // Delegate to the 1D simulation and convert the output to a multi-dimensional
        // representation (each scalar is wrapped in a Vec).
        let (survival_counts, energy_values, values_no_ln, active_walkers_1d,
            final_locations_1d, walker_history_1d) =
            run(Some(h_x), Some(h_tau), Some(n_walkers), Some(max_steps), Some(potential_number))?;
        let active_walkers =
            active_walkers_1d.into_iter().map(|x| vec![x]).collect::<Vec<_>>();
        let final_locations = final_locations_1d
            .into_iter()
            .map(|(x, step)| (vec![x], step))
            .collect::<Vec<_>>();
        let walker_history = walker_history_1d
            .into_iter()
            .map(|history| history.into_iter().map(|x| vec![x]).collect())
            .collect::<Vec<_>>();
        return Ok((
            survival_counts,
            energy_values,
            values_no_ln,
            active_walkers,
            final_locations,
            walker_history,
        ));
    }

    // For n_dim > 1 use the multi-dimensional simulation.
    let result = run_walkers_multi(n_walkers, max_steps, h_x, h_tau, potential_number, n_dim);
    let energy = estimate_ground_state_energy(&result.survival_counts, delta_tau_steps, h_tau);
    // let values = energy.values.iter().map(|&x| x/n_dim as f64).collect::<Vec<_>>();
    // let values_no_ln = energy.values_no_ln.iter().map(|&x| x/n_dim as f64).collect::<Vec<_>>();

    Ok((
        result.survival_counts,
        energy.values,
        energy.values_no_ln,
        result.active_walkers,
        result.final_locations,
        result.walker_history,
    ))
}

