use anyhow::{Context, Result};
use linreg::linear_regression;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::io::Write;
use std::{fs::File, path::PathBuf};

/// Simulation parameters for random walkers
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub j: i32,
    pub num_walkers: usize,
    pub max_steps: usize,
    pub threads: usize,
    pub output_path: Option<PathBuf>,
    pub seed: Option<u64>,
}

/// Results from the simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub j_squared: f64,
    pub lambda: f64,
    pub lambda_j_squared: f64,
    pub residual: f64,
    pub survival_counts: Vec<usize>,
}

impl SimulationConfig {
    /// Create a new simulation configuration with validation
    pub fn new(
        j: i32,
        num_walkers: usize,
        max_steps: usize,
        threads: usize,
        output_path: Option<PathBuf>,
        seed: Option<u64>,
    ) -> Result<Self> {
        if j <= 0 {
            return Err(anyhow::anyhow!("J must be positive"));
        }
        if num_walkers == 0 {
            return Err(anyhow::anyhow!("num_walkers must be positive"));
        }
        if max_steps == 0 {
            return Err(anyhow::anyhow!("max_steps must be positive"));
        }

        Ok(Self {
            j,
            num_walkers,
            max_steps,
            threads,
            output_path,
            seed,
        })
    }

    /// Run the simulation based on this configuration
    pub fn run(&self) -> Result<SimulationResult> {
        // Configure thread pool if specified
        if self.threads > 0 {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build_global();
        }

        let arrest_steps: Vec<usize> = match self.seed {
            Some(seed) => self.simulate_walkers_seeded(seed),
            None => self.simulate_walkers_parallel(),
        };

        // Build frequency array of arrest steps
        let max_possible_step = self.max_steps + 1;
        let freq = self.count_arrest_frequencies(&arrest_steps, max_possible_step);

        // Compute cumulative survival counts
        let survival_counts = self.compute_survival_counts(&freq);

        // Perform linear regression on log of survival counts
        let (j_squared, lambda, lambda_j_squared, residual) = 
            self.analyze_survival_data(&survival_counts)?;

        // Output survival data to file if specified
        if let Some(ref output_path) = self.output_path {
            self.write_output_file(output_path, &survival_counts)?;
        }

        Ok(SimulationResult {
            j_squared,
            lambda,
            lambda_j_squared,
            residual,
            survival_counts,
        })
    }

    fn simulate_walkers_parallel(&self) -> Vec<usize> {
        (0..self.num_walkers)
            .into_par_iter()
            .map_init(
                || rand::rng(),
                |rng, _| {
                    let mut position = 0i32;
                    for step in 1..=self.max_steps {
                        // Move left or right
                        let step_dir: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
                        position += step_dir;

                        // Check if arrested
                        if position.abs() >= self.j {
                            return Some(step);
                        }
                    }
                    None
                },
            )
            .flatten()
            .collect()
    }

    fn simulate_walkers_seeded(&self, seed: u64) -> Vec<usize> {
        let mut rng = Pcg64::seed_from_u64(seed);
        (0..self.num_walkers)
            .filter_map(|_| {
                let mut position = 0i32;
                for step in 1..=self.max_steps {
                    // Move left or right
                    let step_dir: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
                    position += step_dir;

                    // Check if arrested
                    if position.abs() >= self.j {
                        return Some(step);
                    }
                }
                None
            })
            .collect()
    }

    fn count_arrest_frequencies(&self, arrest_steps: &[usize], max_possible_step: usize) -> Vec<usize> {
        let mut freq = vec![0; max_possible_step + 1];
        for &step in arrest_steps {
            if step <= max_possible_step {
                freq[step] += 1;
            }
        }
        freq
    }

    fn compute_survival_counts(&self, freq: &[usize]) -> Vec<usize> {
        let mut survival_counts = vec![0; self.max_steps + 1];
        survival_counts[self.max_steps] = freq[self.max_steps + 1];
        for n in (0..self.max_steps).rev() {
            survival_counts[n] = survival_counts[n + 1] + freq[n + 1];
        }
        survival_counts
    }

    fn analyze_survival_data(&self, survival_counts: &[usize]) -> Result<(f64, f64, f64, f64)> {
        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        for (n, &count) in survival_counts.iter().enumerate() {
            if count > 0 {
                y_values.push((count as f64).ln());
                x_values.push(n as f64);
            }
        }

        let (slope, _intercept): (f64, f64) = linear_regression(&x_values, &y_values).unwrap();
        
        let lambda = -slope;
        let j_squared = (self.j as f64).powi(2);
        let lambda_j_squared = lambda * j_squared;

        // Theoretical value for comparison
        let theoretical = std::f64::consts::PI.powi(2) / 8.0;
        let residual = (lambda_j_squared - theoretical).abs();

        Ok((j_squared, lambda, lambda_j_squared, residual))
    }

    fn write_output_file(&self, output_path: &PathBuf, survival_counts: &[usize]) -> Result<()> {
        let mut file = File::create(output_path)
            .with_context(|| format!("Failed to create output file at {:?}", output_path))?;
        
        writeln!(file, "n\tk(n)\tln_k(n)")?;
        
        for (n, &count) in survival_counts.iter().enumerate() {
            if count > 0 {
                writeln!(file, "{}\t{}\t{}", n, count, (count as f64).ln())?;
            }
        }
        
        Ok(())
    }
}

/// Legacy function for backward compatibility
pub fn sim_walkers(num_walkers: usize, max_steps: usize, j: i32) -> Vec<Option<usize>> {
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
                arrest_step
            },
        )
        .collect()
}

/// Legacy function for backward compatibility
pub fn run(
    j: i32,
    num_walkers: usize,
    max_steps: usize,
    threads: usize,
    output: Option<PathBuf>,
    seed: u64,
) -> Result<(f64, f64, f64, f64)> {
    let config = SimulationConfig::new(
        j,
        num_walkers,
        max_steps,
        threads,
        output,
        if seed == 0 { None } else { Some(seed) },
    )?;
    
    let result = config.run()?;
    
    Ok((
        result.j_squared,
        result.lambda,
        result.lambda_j_squared,
        result.residual,
    ))
}

