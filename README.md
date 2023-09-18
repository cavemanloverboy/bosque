# Bosque

###### *"The clearest way into the Universe is through a forest wilderness."* – John Muir
-------

## What is it?
Bosque (bosh-keh, spanish for "forest") is a fast, parallel **in-place** kDTree Library available for Rust, C, Python, and Julia. This library is intended to be an improvement over [FNNTW](https://github.com/cavemanloverboy/fnntw). It achieves its high performance through similar avenues such as parallel builds and queries, but achieves superior performance by mutating the original data (hence, inplace kDTree). By using a select algorithm to partition the original dataset into buckets that are contiguous within its original allocated buffer, bosque `acheives` incredible cache locality. This same algorithmic choice, along with a round robin dimension spit, results in a kDTree that can be stored with no additional metadata; after building a tree, you can always get the left and right side of a bucket by checking how many times you've split the tree in half so far. Furthermore, this implies that pre-built trees can be zero-copy deserialized (on and across systems with the same endianness) in a matter of nanoseconds.

**At present, since cosmology is the primary use case, the library only supports 3D data. However, it can easily be generalized. The author intends to make the code generic over all dimensions, but feel free to submit an issue + pull request if you want this ASAP!**

-------


## Getting started
Because this is a Rust library, we show an example in Rust here. This example can be found at `examples/rust/simple.rs`. In the `examples` directory, there are examples for C, Python, and Julia.
```rust
// uses the `rand` crate.

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
```


## Performance
Using the same benchmark as in FNNTW, we extend the list. We exclude the original `kiddo` benchmark, as there is now a much faster version 2 of `kiddo` -- which we highly recommend for use cases that cannot be in-place or thatrequire other query functionality e.g. `nearest_within` which `bosque` does not offer.

From FNNTW:
> We use
> - A mock dataset of 100,000 uniform random points in the unit cube.
> - A query set of 1,000,000 uniform random points in the unit cube.
>
> Over 100 realizations of the datasets and query points, the following results are obtained for the average build (serial) and 1NN query times on an AMD EPYC 7502P using 48 threads. The results are sorted by the combined build and query time.

|  Code | Build (ms)| Query (ms) | Total (ms) |
|---|---|---|---|
| **Bosque**| **X** | **X** | **X** |
| FNNTW | 12 | 22 | 34 |
| pykdtree (python)| 12 | 35 | 47  |
| nabo-rs (rust)| 25 | 30  | 55 |
| Scipy's cKDTree (python) | 31 | 38 | 69 |