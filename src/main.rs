mod walkv1;
mod walkv2;

fn main() { 
    match walkv2::run(None, None, None, None) {
        Ok(_) => println!("Done!"),
        Err(e) => println!("Error: {}", e),
    }
}
