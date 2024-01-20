mod constants;
mod model;

fn get_weights() -> model::Weights {
    todo!()
}

fn main() {
    println!("starting...");
    let mut state = model::State::<1>::new();
    let weights = get_weights();
    state.update(&[0], &weights);
    println!("{:?}", state.logits());
}
