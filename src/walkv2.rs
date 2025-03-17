use rand::prelude::*;
use anyhow::Result;

fn potential(x: f64) -> f64 {
    0.5 * x * x
}

fn potential1(x: f64) -> f64 {
    x * x 
}

fn run_walkers(n_walkers: usize, max_steps: usize, h_x: f64, h_tau: f64, potential_number: usize) -> (Vec<usize>, Vec<f64>, Vec<(f64, usize)>, Vec<Vec<f64>>) {
    let potential = match potential_number {
        1 => {
            potential
        }
        2 => {
            potential1
        }
        _ => panic!("Invalid potential function")
    };
    let mut rng = rand::rng();
    let mut active_walkers = vec![0.0; n_walkers];
    let mut survival_counts = Vec::with_capacity(max_steps);
    let mut final_locations: Vec<(f64, usize)> = Vec::with_capacity(n_walkers);
    let mut active_walkers_at_all_steps: Vec<Vec<f64>> = Vec::new();

    for step in 0..max_steps {
        let mut next_walkers = Vec::with_capacity(active_walkers.len());
        
        for &x in &active_walkers {
            let x_new = x + if rng.random_bool(0.5) { h_x } else { -h_x };
            let a = h_tau * potential(x_new);
            if rng.random::<f64>() >= a {
                next_walkers.push(x_new);
            }
            else {
                final_locations.push((x_new, step));
            }
        }
        survival_counts.push(next_walkers.len());
        active_walkers = next_walkers.clone();
        active_walkers_at_all_steps.push(next_walkers.clone());
    }
    (survival_counts, active_walkers, final_locations, active_walkers_at_all_steps)
}

fn e0_estimate(survival_counts: &Vec<usize>, delta_tau_steps: usize, h_tau: f64) -> (Vec<f64>, Vec<f64>) {
    (0..(survival_counts.len() - delta_tau_steps))
        .map(|i| {
            let current = survival_counts[i] as f64;
            let future = survival_counts[i + delta_tau_steps] as f64;
            
            if current > 0.0 && future > 0.0 {
                let ratio = future / current;
                (-ratio.ln() / (delta_tau_steps as f64 * h_tau),
                ratio / (delta_tau_steps as f64 * h_tau))
            } else {
                (0.0, 0.0)
            }
        }).collect()
}

pub fn run(h_x: Option<f64>, h_tau: Option<f64>, num_walkers: Option<usize>, max_steps: Option<usize>, potential_number: Option<usize>) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<(f64, usize)>, Vec<Vec<f64>>)> {
    // Simulation parameters
    let h_x = h_x.unwrap_or(0.25);
    let h_tau = h_tau.unwrap_or(h_x * h_x);
    let n_walkers = num_walkers.unwrap_or(10_000);
    let max_steps = max_steps.unwrap_or(32);
    let delta_tau_steps = 10; // Δτ = 10 * h_tau = 0.625
    let potential_number = potential_number.unwrap_or(1);

    // Run walker
    let (survival_counts, active_walkers, final_walkers, active_walkers_at_all_steps) = run_walkers(n_walkers, max_steps, h_x, h_tau, potential_number);

    // Calculate ground state energy estimates
    // println!("τ\tE0 Estimate");
    // for i in 0..(survival_counts.len() - delta_tau_steps) {
    //     let current = survival_counts[i] as f64;
    //     let future = survival_counts[i + delta_tau_steps] as f64;
        
    //     if current > 0.0 && future > 0.0 {
    //         let ratio = future / current;
    //         let e0_est = -ratio.ln() / (delta_tau_steps as f64 * h_tau);
    //         println!("{:.2}\t{:.4}", i as f64 * h_tau, e0_est);
    //     }
    // };

    // Calculate ground state energy estimates
    let e0_estimates = e0_estimate(&survival_counts, delta_tau_steps, h_tau);

    Ok((survival_counts, e0_estimates.0, e0_estimates.1, active_walkers, final_walkers, active_walkers_at_all_steps))
}
