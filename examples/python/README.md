# Bosque in Python
-------
We use `pyo3` to expose the Rust library to Python. To set it up from source follow these instructions:

1) First get maturin using `pip install maturin`
2) Navigate to the `py/` directory (the `bosque-py` crate) and run `maturin develop --release`[^1].
3) Your python environment should now have `pybosque`!



# Footnotes
[^1]:You can include `RUSTFLAGS="-C target-cpu=native"` at the beginning of this command, i.e. `RUSTFLAGS="-C target-cpu=native" maturin develop --release` to give the compiler hints to use cpu intrinsics (e.g. avx/avx2) if your machine supports them.
