use rand::prelude::*;
use std::f64::consts::PI;

pub fn main() {
    // Simulation parameters
    let h_x = 0.25;          // Spatial step size
    let h_tau = h_x * h_x;   // Time step size (h_tau = h_x²)
    let n_walkers = 10_000;  // Number of initial walkers
    let max_steps = 32;       // Corresponds to τ = 2.0 (32 * 0.0625)
    let delta_tau_steps = 10; // Δτ = 10 * h_tau = 0.625

    // Potential function (SHO)
    let potential = |x: f64| 0.5 * x * x;

    // Initialize random number generator and walkers
    let mut rng = rand::rng();
    let mut active_walkers = vec![0.0; n_walkers];
    let mut survival_counts = Vec::with_capacity(max_steps);
    let mut tau_2_positions = Vec::new();

    // Main simulation loop
    for step in 0..max_steps {
        let mut next_walkers = Vec::with_capacity(active_walkers.len());
        
        for &x in &active_walkers {
            // Random step direction
            let x_new = x + if rng.random_bool(0.5) { h_x } else { -h_x };
            
            // Absorption probability
            let a = h_tau * potential(x_new);
            if rng.random::<f64>() >= a {
                next_walkers.push(x_new);
            }
        }
        
        // Record survival count and positions at τ=2.0
        if step == max_steps - 1 {
            tau_2_positions = next_walkers.clone();
        }
        survival_counts.push(next_walkers.len());
        active_walkers = next_walkers;
    }

    // Calculate ground state energy estimates
    println!("τ\tE0 Estimate");
    for i in 0..(survival_counts.len() - delta_tau_steps) {
        let current = survival_counts[i] as f64;
        let future = survival_counts[i + delta_tau_steps] as f64;
        
        if current > 0.0 && future > 0.0 {
            let ratio = future / current;
            let e0_est = -ratio.ln() / (delta_tau_steps as f64 * h_tau);
            println!("{:.2}\t{:.4}", i as f64 * h_tau, e0_est);
        }
    }

    // Generate wavefunction histogram at τ=2.0
    let bin_width = 0.5;
    let (min_x, max_x) = (-5.0, 5.0);
    let num_bins = ((max_x - min_x) / bin_width) as usize;
    let mut histogram = vec![0; num_bins];

    for &x in &tau_2_positions {
        let bin = ((x - min_x) / bin_width) as usize;
        if bin < num_bins {
            histogram[bin] += 1;
        }
    }

    // Normalize and print histogram
    println!("\nWavefunction histogram at τ=2.0:");
    println!("Bin Center\tDensity\t\tExact");
    let total = tau_2_positions.len() as f64;
    for (i, &count) in histogram.iter().enumerate() {
        let center = min_x + (i as f64 + 0.5) * bin_width;
        let density = (count as f64) / total / bin_width;
        let exact = (-center.powi(2)).exp() / PI.sqrt();
        println!("{:.2}\t\t{:.4}\t\t{:.4}", center, density, exact);
    }
}
