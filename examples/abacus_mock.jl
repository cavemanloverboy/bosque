include("api.jl")
const libbosque = joinpath(@__DIR__, "..", "target", "release", "libbosque.dylib")

# Compress + Decompress round trip
xv = decompress(compress(0.25, 750))
println("round trip $xv")

# Make random data points
# Set the seed for reproducibility (optional)
using Random
Random.seed!(123)

# Define the number of points
num_points = 1_000_000

# Generate the random points
random_points = (rand(num_points) .- 0.5) .* 1.0

# Display the first few points
@show random_points[1:10]