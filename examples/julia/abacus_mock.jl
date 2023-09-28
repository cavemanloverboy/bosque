include("api.jl")
const libbosque = joinpath(@__DIR__, "..", "..", "target", "release", "libbosque.dylib")
# on linux:
# const libbosque = joinpath(@__DIR__, "..", "..", "target", "release", "libbosque.so")


# Compress + Decompress round trip
xv = decompress(compress(0.25, 750))
println("\nround trip (0.25, 750)-> -> $xv\n")

# Make random data points
# Set the seed for reproducibility (optional)
using Random
Random.seed!(123)

# Define the number of data points
num_points = 100_000

# Generate the random data points (3D as flat buffer)
data_points = Cfloat.(rand(num_points * 3) .- 0.5f0)
indices = collect(0:num_points)

# Build the tree
println("\nBuilding tree with $num_points elements\n")
@time tree = construct_tree_f32(data_points, num_points, indices)

# Generate random query points (3D as flat buffer)
num_query = 1_000_000
queries = Cfloat.(rand(num_query * 3) .- 0.5f0)

# Query the tree
# @time result = query_f32_nearest(data_points, num_points, queries, num_query)
println("\nQuerying tree with $num_query queries\n")
@time result = query_f32_nearest_parallel(data_points, num_points, queries, num_query)
result = unsafe_load(result)
@show result.dist_sq
