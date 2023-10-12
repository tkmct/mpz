use mpz_circuits::{evaluate, trace, CircuitBuilder};

fn main() {
    let builder = CircuitBuilder::new();
    let a = builder.add_input::<bool>();
    let b = builder.add_input::<bool>();

    let mut state = builder.state().borrow_mut();

    println!("{:?}", builder.state());
}
