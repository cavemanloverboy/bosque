# Using Bosque in C and C++
When running `cargo build --release`, the custom `build.rs` writes C and C++ header files for the Rust library at `examples/C/bosque.{h, hpp}`.

We provide a simple Makefile at the root of the repository which can compile the included `abacus_mock.c` example code with `make` followed by `make mock`. By default, this will use `RUSTFLAGS="-C target-cpu=native` and `-march=native`, which should provide the compiler with hints to use cpu intrinsics (e.g. avx/avx2 instructions) if your machine supports them.