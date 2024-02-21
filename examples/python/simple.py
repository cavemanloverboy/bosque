from bosque_py import Tree
import numpy as np
from copy import deepcopy
from time import perf_counter as time

# Bucket size is fixed at 32, so we need larger value
DATA = 64
QUERY = 4

# Bosque only works for d=3 right now
DIM = 3
K = 2
K_SPARSE = [1, 2, 4]

# Generate data
data = np.random.uniform(size=(DATA, DIM))
data_copy = deepcopy(data)
query = np.random.uniform(size=(QUERY, DIM))
idxs = np.arange(DATA, dtype=np.uint32)

# Build tree
tree = Tree(data, idxs)

# Query
r, ids = tree.query(query, K, [0, 1])

# Original ids
og_ids = idxs[ids]

# Verify original ids
# These point pairs should be the same
# [0, 0] --> 1NN of first query point
print(query[0], data[ids[0, 0]])
print(query[0], data_copy[og_ids[0, 0]])

# Consequently, the distance should be the same
r0 = np.sqrt(np.sum((query[0] - data[ids[0, 0]])**2))
d = np.sqrt(np.sum((query[0] - data_copy[og_ids[0, 0]])**2))
print(d, r0, r[0, 0])
