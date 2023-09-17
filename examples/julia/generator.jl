
#
# This script generates `api.jl` in the current folder
# that can be used as a starting point for `JuliaPackageWithRustDep.jl/api.jl`.
#
# Run this script with the following command:
# `julia generator.jl`
#
import Pkg;
Pkg.add("Clang");
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Clang.Generators
using Clang.LibClang.Clang_jll

# We are in CARGO_MANIFEST_DIR/examples/julia, 
cargo_dir = joinpath(@__DIR__, "..", "..")

# C headers are always built
include_dir = joinpath(cargo_dir, "examples/c/")
print(cargo_dir)
@assert isdir(include_dir)

# wrapper generator options
options = load_options(joinpath(cargo_dir, "examples", "julia", "generator.toml"))

# add compiler flags, e.g. "-DXXXXXXXXX"
args = get_default_args()
push!(args, "-I$include_dir")

headers = [joinpath(include_dir, header) for header in readdir(include_dir) if endswith(header, ".h")]
# there is also an experimental `detect_headers` function for auto-detecting top-level headers in the directory
# headers = detect_headers(clang_dir, args)

# create context
ctx = create_context(headers, args, options)

# run generator
build!(ctx)