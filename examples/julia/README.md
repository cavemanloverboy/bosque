# Bindings for Julia

- You must first build the dylib using `cargo build --release`. For SIMD and other intrinsics, do `RUSTFLAGS="-C target-cpu=native" cargo build --release`. Note this makes the dylib less portable.
- Then build the `api.jl` using `generator.jl` 
- Finally, use the library as in `abacus_mock.jl`. You must import the `api.jl` and the `libbosque.dylib`.