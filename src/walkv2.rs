use anyhow::Result;
use rand::prelude::*;

/// Harmonic oscillator potential: 0.5 * x^2
fn potential_harmonic(x: f64) -> f64 {
    0.5 * x * x
}

/// Alternative potential: x^2
fn potential_linear(x: f64) -> f64 {
    x
}

/// Represents simulation results from diffusion Monte Carlo
struct SimulationResult {
    /// Number of surviving walkers at each step
    survival_counts: Vec<usize>,
    /// Final positions of surviving walkers
    active_walkers: Vec<f64>,
    /// Positions and steps at which walkers were terminated
    final_locations: Vec<(f64, usize)>,
    /// Positions of all active walkers at each step
    walker_history: Vec<Vec<f64>>,
}

/// Represents ground state energy estimates
struct EnergyEstimate {
    /// Energy estimates
    values: Vec<f64>,
    /// Uncertainties in energy estimates
    uncertainties: Vec<f64>,
}

/// Run diffusion Monte Carlo simulation with walkers
fn run_walkers(
    n_walkers: usize, 
    max_steps: usize, 
    h_x: f64, 
    h_tau: f64, 
    potential_number: usize,
) -> SimulationResult {
    let potential = match potential_number {
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

/// Calculate ground state energy estimates
fn estimate_ground_state_energy(
    survival_counts: &[usize], 
    delta_tau_steps: usize, 
    h_tau: f64,
) -> EnergyEstimate {
    let mut values = Vec::new();
    let mut uncertainties = Vec::new();
    
    for i in 0..(survival_counts.len() - delta_tau_steps) {
        let current = survival_counts[i] as f64;
        let future = survival_counts[i + delta_tau_steps] as f64;
        
        if current > 0.0 && future > 0.0 {
            let ratio = future / current;
            values.push(-ratio.ln() / (delta_tau_steps as f64 * h_tau));
            uncertainties.push(ratio / (delta_tau_steps as f64 * h_tau));
        } else {
            values.push(0.0);
            uncertainties.push(0.0);
        }
    }
    
    EnergyEstimate {
        values,
        uncertainties,
    }
}

/// Run a full Monte Carlo diffusion simulation
pub fn run(
    h_x: Option<f64>, 
    h_tau: Option<f64>, 
    num_walkers: Option<usize>, 
    max_steps: Option<usize>, 
    potential_number: Option<usize>,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<(f64, usize)>, Vec<Vec<f64>>)> {
    // Simulation parameters with defaults
    let h_x = h_x.unwrap_or(0.25);
    let h_tau = h_tau.unwrap_or(h_x * h_x);
    let n_walkers = num_walkers.unwrap_or(10_000);
    let max_steps = max_steps.unwrap_or(32);
    let delta_tau_steps = 10; // Δτ = 10 * h_tau = 0.625
    let potential_number = potential_number.unwrap_or(1);

    // Run simulation
    let result = run_walkers(n_walkers, max_steps, h_x, h_tau, potential_number);
    
    // Calculate ground state energy estimates
    let energy = estimate_ground_state_energy(&result.survival_counts, delta_tau_steps, h_tau);

    // Return data in the original format for backward compatibility
    Ok((
        result.survival_counts, 
        energy.values, 
        energy.uncertainties, 
        result.active_walkers, 
        result.final_locations, 
        result.walker_history,
    ))
}

