use rayon::prelude::*;
use rand::Rng;

struct Walker {
    position: i32,
    arrest_step: Option<usize>,
}

impl Walker {
    fn new() -> Self {
        Self {
            position: 0,
            arrest_step: None,
        }
    }

    fn step(&mut self, rng: &mut impl Rng, j: i32) {
        let step_dir: i32 = if rng.random_bool(0.5) { 1 } else { -1 };
        self.position += step_dir;
        if self.position.abs() >= j {
            self.arrest_step = Some(self.position.abs() as usize);
        }
    }
}

pub fn walk(j: i32, num_walkers: usize, max_steps: usize) -> Vec<usize> {
    (0..num_walkers)
        .into_par_iter()
        .map_init(|| rand::rng(), |mut rng, _| {
            let mut walker = Walker::new();
            for _ in 1..=max_steps {
                walker.step(&mut rng, j);
                if walker.arrest_step.is_some() {
                    break;
                }
            }
            walker.arrest_step.unwrap_or(max_steps + 1)
        })
        .collect()
}
