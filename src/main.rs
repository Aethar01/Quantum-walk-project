use clap::Parser;
use std::path::PathBuf;

mod walk;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Number of random walks to simulate
    #[arg(short, long, default_value_t = 1000)]
    num_walkers: usize,

    /// Lattice size J (boundary at ±J)
    #[arg(short, long, default_value_t = 8)]
    j: i32,

    /// Maximum number of steps per walk
    #[arg(short('m'), long, default_value_t = 100)]
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
    let args = Args::parse();
    if let Err(e) = walk::run(args.j, args.num_walkers, args.max_steps, args.threads, args.output, args.seed) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
