use clap::Parser;
use std::path::PathBuf;

mod walk;
mod walkv2;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Number of random walks to simulate
    #[arg(short, long, default_value_t = 1000)]
    num_walkers: usize,

    /// Lattice size J (boundary at ±J)
    #[arg(short, long, default_value_t = 17)]
    j: i32,

    /// Maximum number of steps per walk
    #[arg(short('m'), long, default_value_t = 1000)]
    max_steps: usize,

    /// Number of threads to use (0 for automatic)
    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    /// Output file for survival data (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Seed for random number generators
    /// If 0 or unspecified, a random seed is used
    #[arg(short, long, default_value_t = 0)]
    seed: u64,
}

fn main() { 
    // let args = Args::parse();
    // match walk::run(args.j, args.num_walkers, args.max_steps, args.threads, args.output, args.seed) {
    //     Ok((_, lambda_j_sq, residual)) => {
    //         println!("Computed λJ²: {:.4}", lambda_j_sq);
    //         println!("Theoretical π²/8: {:.4}", std::f64::consts::PI.powi(2) / 8.0);
    //         println!("Residual: {:.2}%", residual);
    //     }
    //     Err(e) => {
    //         eprintln!("Error: {}", e);
    //         std::process::exit(1);
    //     }
    // }
    walkv2::main();
    walkv3::main();
}
