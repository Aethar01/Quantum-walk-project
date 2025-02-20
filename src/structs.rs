use clap::Parser;
use std::path::PathBuf;

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
}
