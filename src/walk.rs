use anyhow::{Context, Result};
use linreg::linear_regression;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::io::Write;
use std::{fs::File, path::PathBuf};

fn sim_walkers(num_walkers: usize, max_steps: usize, j: i32) -> Vec<usize> {
    (0..num_walkers)
        .into_par_iter()
        .map_init(
            || rand::rng(),
            |rng, _| {
                let mut position = 0i32;
                let mut arrest_step = None;
                for step in 1..=max_steps {
                    // Move left or right
                    let step_dir: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
                    position += step_dir;

                    // Check if arrested
                    if position.abs() >= j {
                        arrest_step = Some(step);
                        break;
                    }
                }
                arrest_step.unwrap_or(max_steps + 1)
            },
        )
        .collect()
}

/// NOT MULTITHREADED
fn sim_walkers_seeded(num_walkers: usize, max_steps: usize, j: i32, seed: u64) -> Vec<usize> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..num_walkers)
        .map(|_| {
            let mut position = 0i32;
            let mut arrest_step = None;
            for step in 1..=max_steps {
                // Move left or right
                let step_dir: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
                position += step_dir;

                // Check if arrested
                if position.abs() >= j {
                    arrest_step = Some(step);
                    break;
                }
            }
            arrest_step.unwrap_or(max_steps + 1)
        })
        .collect()
}

fn freq_array_of_arrest_steps(max_possible_step: usize, arrest_steps: Vec<usize>) -> Vec<usize> {
    let mut freq = vec![0; max_possible_step + 1];
    for &step in &arrest_steps {
        if step <= max_possible_step {
            freq[step] += 1;
        }
    }
    freq
}

/// Returns (j_sq, lambda_j_sq, residual)
pub fn run(
    j: i32,
    num_walkers: usize,
    max_steps: usize,
    threads: usize,
    output: Option<PathBuf>,
    seed: u64,
) -> Result<(f64, f64, f64)> {
    // Validate inputs
    if j <= 0 {
        return Err(anyhow::anyhow!("J must be positive"));
    }
    if num_walkers <= 0 {
        return Err(anyhow::anyhow!("num_walkers must be positive"));
    }
    if max_steps <= 0 {
        return Err(anyhow::anyhow!("max_steps must be positive"));
    }

    // Set number of threads
    if threads > 0 {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            Ok(_) => (),
            Err(_) => (),
        }
    }

    let arrest_steps: Vec<usize> = {
        if seed == 0 {
            sim_walkers(num_walkers, max_steps, j)
        } else {
            sim_walkers_seeded(num_walkers, max_steps, j, seed)
        }
    };

    // Build frequency array of arrest steps
    let max_possible_step: usize = max_steps + 1;
    let freq: Vec<usize> = freq_array_of_arrest_steps(max_possible_step, arrest_steps);

    // Compute cumulative survival counts
    let mut cum_sum = vec![0; max_steps + 1];
    cum_sum[max_steps] = freq[max_steps + 1];
    for n in (0..max_steps).rev() {
        cum_sum[n] = cum_sum[n + 1] + freq[n + 1];
    }

    let mut xcol: Vec<f64> = Vec::new();
    let mut ycol: Vec<f64> = Vec::new();

    for n in 0..=max_steps {
        let k = cum_sum[n];
        if k == 0 {
            continue;
        }
        let y = (k as f64).ln();
        let x = n as f64;
        ycol.push(y);
        xcol.push(x);
    }

    let (slope, _): (f64, _) = linear_regression(xcol.as_slice(), ycol.as_slice()).unwrap();
    let lambda = -slope;

    // Compute lambda * J^2
    let j_sq = (j as f64).powi(2);
    let lambda_j_sq = lambda * j_sq;

    // Theoretical value for comparison
    let theoretical = std::f64::consts::PI.powi(2) / 8.0;
    let residual = (lambda_j_sq - theoretical).abs() / theoretical * 100.0;

    // Output survival data to file if specified
    write_output_file(output, cum_sum, max_steps)?;

    Ok((j_sq, lambda_j_sq, residual))
}

fn write_output_file(output: Option<PathBuf>, cum_sum: Vec<usize>, max_steps: usize) -> Result<()> {
    if let Some(output_path) = output {
        let mut file = File::create(output_path).context("Failed to create output file")?;
        writeln!(file, "n\tk(n)\tln_k(n)").context("Failed to write header")?;
        for n in 0..=max_steps {
            let k = cum_sum[n];
            if k > 0 {
                writeln!(file, "{}\t{}\t{}", n, k, (k as f64).ln())
                    .context("Failed to write data line")?;
            }
        }
    }
    Ok(())
}
