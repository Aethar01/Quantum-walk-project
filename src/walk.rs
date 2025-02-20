use rand::Rng;
use rayon::prelude::*;
use std::{fs::File, path::PathBuf};
use std::io::Write;
use anyhow::{Context, Result};

fn sim_walkers(num_walkers: usize, max_steps: usize, j: i32) -> Vec<usize> {
    (0..num_walkers)
        .into_par_iter()
        .map_init(
            || rand::rng(),
            |rng, _| {
                let mut position = 0i32;
                let mut arrest_step = None;
                for step in 1..= max_steps {
                    // Move left or right
                    let step_dir: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
                    position += step_dir;

                    // Check if arrested
                    if position.abs() > j {
                        arrest_step = Some(step);
                        break;
                    }
                }
                arrest_step.unwrap_or(max_steps + 1)
            },
        )
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

pub fn run(j: i32, num_walkers: usize, max_steps: usize, threads: usize, output: Option<PathBuf>) -> Result<f64> {

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

    // Set number of threads if specified
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("Failed to build thread pool")?;
    }

    // Simulate all walkers in parallel
    let arrest_steps: Vec<usize> = sim_walkers(num_walkers, max_steps, j);

    // Build frequency array of arrest steps
    let max_possible_step: usize = max_steps + 1;
    let freq: Vec<usize> = freq_array_of_arrest_steps(max_possible_step, arrest_steps);

    // Compute cumulative survival counts
    let mut cum_sum = vec![0; max_steps + 1];
    cum_sum[max_steps] = freq[max_steps + 1];
    for n in (0..max_steps).rev() {
        cum_sum[n] = cum_sum[n + 1] + freq[n + 1];
    }

    // Collect data points for linear regression
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut n_points = 0;

    for n in 0..=max_steps {
        let k = cum_sum[n];
        if k == 0 {
            continue;
        }
        let y = (k as f64).ln();
        let x = n as f64;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        n_points += 1;
    }

    // Check if we have enough data points for regression
    if n_points < 2 {
        return Err(anyhow::anyhow!(
            "Insufficient data points (n_points={}) for regression", n_points
        ));
    }

    // Compute slope (lambda) and intercept
    let denominator = (n_points as f64) * sum_x2 - sum_x * sum_x;
    if denominator == 0.0 {
        return Err(anyhow::anyhow!(
            "Regression denominator is zero; insufficient data variation"
        ));
    }
    let slope = ((n_points as f64) * sum_xy - sum_x * sum_y) / denominator;
    let lambda = -slope;

    // Compute lambda * J^2
    let j_sq = (j as f64).powi(2);
    let lambda_j_sq = lambda * j_sq;

    // Theoretical value for comparison
    let theoretical = std::f64::consts::PI.powi(2) / 8.0;

    println!("Computed λJ²: {:.4}", lambda_j_sq);
    println!("Theoretical π²/8: {:.4}", theoretical);
    println!(
        "Residual: {:.2}%",
        (lambda_j_sq - theoretical).abs() / theoretical * 100.0
    );

    // Output survival data to file if specified
    write_output_file(output, cum_sum, max_steps)?;

    Ok(lambda_j_sq)
}

fn write_output_file(output: Option<PathBuf>, cum_sum: Vec<usize>, max_steps: usize) -> Result<()> {
    if let Some(output_path) = output {
        let mut file = File::create(output_path).context("Failed to create output file")?;
        writeln!(file, "n\tk(n)\tln_k(n)").context("Failed to write header")?;
        for n in 0..= max_steps {
            let k = cum_sum[n];
            if k > 0 {
                writeln!(file, "{}\t{}\t{}", n, k, (k as f64).ln())
                    .context("Failed to write data line")?;
            }
        }
    }
    Ok(())
}
