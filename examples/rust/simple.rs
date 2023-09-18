fn main() {
    // Initialize some data in [0, 1]^3
    let mut pos: Vec<[f64; 3]> = (0..100_000)
        .map(|_| [(); 3].map(|_| rand::random::<f64>()))
        .collect();

    // Build tree in-place!
    bosque::tree::build_tree(&mut pos);

    // Query the tree
    let query = [0.5, 0.5, 0.5];
    let (dist_metric, id) = bosque::tree::nearest_one(&pos, &query);

    // Note that there is a 'sqrt-dist' feature for
    // returning square euclidean or euclidean distance!
    let dist = if cfg!(feature = "sqrt-dist") {
        dist_metric
    } else {
        dist_metric.sqrt()
    };

    println!("closest point is {dist:.2e} units away, corresponding to data point #{id}");
}
