use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;

/// Harmonic oscillator potential V(x) = x²/2
fn potential(x: f64) -> f64 {
    0.5 * x * x
}

/// Run quantum Monte Carlo simulation for a harmonic oscillator
#[pyfunction]
pub fn run_qmc_simulation(
    n0: usize,
    mcs: usize,
    ds: f64,
    w0: f64,
    dt_factor: usize,
) -> PyResult<HashMap<String, Vec<Vec<f64>>>> {
    // Time step from diffusion relation
    let dt: f64 = ds * ds / dt_factor as f64;
    
    // Initialize random number generator
    let mut rng = rand::rng();
    
    // Initialize walkers with random gaussian distribution positions within w0 of origin
    let mut walkers: Vec<f64> = (0..n0)
        .map(|_| {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            r * theta.cos() * w0 / 2.0  // Gaussian distribution
        })
        .collect();
    
    // Initialize energy tracking
    let mut e_sum: f64 = 0.0;
    let mut v_ref: f64 = 0.0;
    let mut energy_data = Vec::new();
    let mut vref_data = Vec::new();
    let mut walker_count_data = Vec::new();
    
    // Store walker positions at save steps
    let mut all_walker_positions: HashMap<usize, Vec<f64>> = HashMap::new();
    
    // Calculate initial reference energy
    v_ref = walkers.iter().map(|&x| potential(x)).sum::<f64>() / walkers.len() as f64;
    
    // Main simulation loop
    for imcs in 1..=mcs {
        let n_before = walkers.len();
        
        // Process each walker
        let mut i = 0;
        while i < walkers.len() {
            // Random move (left or right)
            if rng.random::<bool>() {
                walkers[i] += ds;
            } else {
                walkers[i] -= ds;
            }
            
            // Calculate potential and potential difference
            let v = potential(walkers[i]);
            let dv = v - v_ref;
            
            // Birth/death process
            let r = rng.random::<f64>();
            let birth_death_prob = (dv * dt).abs().min(0.3);  // Limit max probability

            if dv > 0.0 && r < birth_death_prob {
                // Remove walker
                walkers.swap_remove(i);
                continue; // Don't increment i since we removed a walker
            } else if dv < 0.0 && r < birth_death_prob {
                // Add walker
                walkers.push(walkers[i]);
            }
            
            i += 1;
        }
        
        // Calculate mean potential
        let v_mean = if walkers.is_empty() {
            0.5  // Default to theoretical value if no walkers
        } else {
            walkers.iter().map(|&x| potential(x)).sum::<f64>() / walkers.len() as f64
        };

         // Damping factor for Vref updates (prevents wild oscillations)
        let damping: f64 = 0.01;
        
        // Update reference potential
        let n_after = walkers.len();
        if n_before > 0 {
            let v_ref_update = v_mean - (n_after as f64 - n_before as f64) / (n_before as f64 * dt);
            // Apply damping to prevent wild oscillations
            v_ref = (1.0 - damping) * v_ref + damping * v_ref_update;
        }
        
        // Accumulate energy
        e_sum += v_mean;
        let e_avg = e_sum / imcs as f64;
        
        // Store data for plotting
        energy_data.push(vec![imcs as f64, e_avg]);
        vref_data.push(vec![imcs as f64, v_ref]);
        walker_count_data.push(vec![imcs as f64, walkers.len() as f64]);
        
        // Store walker positions at all steps
        all_walker_positions.insert(imcs, walkers.clone());
    }
    
    // Generate exact solution
    let mut exact_solution = Vec::new();
    for i in 0..200 {
        let xmin = -6.0;
        let xmax = 6.0;
        let x = xmin + (xmax - xmin) * i as f64 / 199.0;
        let psi = std::f64::consts::PI.powf(-0.25) * (-0.5 * x * x).exp();
        exact_solution.push(vec![x, psi]);
    }
    
    // Prepare return data
    let mut result = HashMap::new();
    result.insert("energy".to_string(), energy_data);
    result.insert("vref".to_string(), vref_data);
    result.insert("walker_count".to_string(), walker_count_data);
    result.insert("exact_solution".to_string(), exact_solution);

    for i in 1..=mcs {
        if let Some(positions) = all_walker_positions.get(&i) {
            let positions_vec: Vec<Vec<f64>> = positions.iter().map(|&x| vec![x]).collect();
            result.insert(format!("walkers_{}", i), positions_vec);
        }
    }
    
    Ok(result)
}
