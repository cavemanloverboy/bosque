using CEnum

struct DecompressedPair
    pos::Cfloat
    vel::Cfloat
end

struct CP32
    _0::UInt32
end

const Index = UInt32

struct QueryNearest
    dist_sq::Cfloat
    idx_within::UInt64
end

function compress(position, velocity)
    ccall((:compress, libbosque), UInt32, (Cfloat, Cfloat), position, velocity)
end

function decompress(compressed)
    ccall((:decompress, libbosque), DecompressedPair, (UInt32,), compressed)
end

function construct_compressed_tree(flat_data_ptr, num_points, idxs_ptr)
    ccall((:construct_compressed_tree, libbosque), Cvoid, (Ptr{CP32}, UInt64, Ptr{Index}), flat_data_ptr, num_points, idxs_ptr)
end

function construct_tree_f32(flat_data_ptr, num_points, idxs_ptr)
    ccall((:construct_tree_f32, libbosque), Cvoid, (Ptr{Cfloat}, UInt64, Ptr{Index}), flat_data_ptr, num_points, idxs_ptr)
end

function query_compressed_nearest(flat_data_ptr, num_points, flat_query_ptr, num_queries)
    ccall((:query_compressed_nearest, libbosque), Ptr{QueryNearest}, (Ptr{CP32}, UInt64, Ptr{Cfloat}, UInt64), flat_data_ptr, num_points, flat_query_ptr, num_queries)
end

function query_f32_nearest(flat_data_ptr, num_points, flat_query_ptr, num_queries)
    ccall((:query_f32_nearest, libbosque), Ptr{QueryNearest}, (Ptr{Cfloat}, UInt64, Ptr{Cfloat}, UInt64), flat_data_ptr, num_points, flat_query_ptr, num_queries)
end

function query_compressed_nearest_parallel(flat_data_ptr, num_points, flat_query_ptr, num_queries)
    ccall((:query_compressed_nearest_parallel, libbosque), Ptr{QueryNearest}, (Ptr{CP32}, UInt64, Ptr{Cfloat}, UInt64), flat_data_ptr, num_points, flat_query_ptr, num_queries)
end

function query_f32_nearest_parallel(flat_data_ptr, num_points, flat_query_ptr, num_queries)
    ccall((:query_f32_nearest_parallel, libbosque), Ptr{QueryNearest}, (Ptr{Cfloat}, UInt64, Ptr{Cfloat}, UInt64), flat_data_ptr, num_points, flat_query_ptr, num_queries)
end

const BUCKET_SIZE = 32

