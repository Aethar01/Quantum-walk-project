use std::io::Write;
mod walkv1;
mod walkv2;
mod walkv3;

fn main() { 
    let num_walkers = 10000;
    let max_steps = 100;
    let j = 8;
    let data = walkv1::sim_walkers(num_walkers, max_steps, j).into_iter().flatten().collect::<Vec<_>>();
    // println!("{:?}", data);

    // make histogram
    let counts: Vec<usize> = {
        let mut freq = vec![0; max_steps + 1];
        for &step in &data {
            freq[step] += 1;
        }
        freq
    };

    // print histogram
    let mut lock = std::io::stdout().lock();
    for (i, &count) in counts.iter().enumerate() {
        writeln!(lock, "{}\t{}", i, count).unwrap();
    }
}
