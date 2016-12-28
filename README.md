PowerWalk
=========

Most methods for **Personalized PageRank (PPR)** precompute and store all
accurate PPR vectors, and at query time, return the ones of interest directly.
However, the storage and computation of all accurate PPR vectors can be
prohibitive for large graphs, especially in caching them in memory for
real-time online querying. We propose a distributed framework, **PowerWalk**,
that strikes a better balance between offline indexing and online querying. The
offline indexing attains a fingerprint of the PPR vector of each vertex by
performing billions of "short" random walks in parallel across a cluster of
machines. We prove that our indexing method has an exponential convergence,
achieving the same precision with previous methods using a much smaller number
of random walks. At query time, the new PPR vector is composed by a linear
combination of related fingerprints, in a highly efficient vertex-centric
decomposition manner. Interestingly, the resulting PPR vector is much more
accurate than its offline counterpart because it actually uses more random
walks in its estimation. More importantly, we show that such decomposition for
a batch of queries can be very efficiently processed using a shared
decomposition. Our implementation takes advantage of advanced distributed graph
engines and it outperforms the state-of-the-art algorithms by orders of
magnitude. Particularly, it responses to tens of thousands of queries on graphs
with billions of edges in just a few seconds.

For the detailed description of our algorithm please refer to our
[paper](https://lqhl.me/publication/cikm2016.pdf) on CIKM'16:
> PowerWalk: Scalable Personalized PageRank via Random Walks with Vertex-Centric Decomposition  
> Qin Liu, Zhenguo Li, John C.S. Lui, Jiefeng Cheng.  
> The 25th ACM International Conference on Information and Knowledge Management (CIKM), 2016

This repository contains our implementation of the online querying algorithm
built atop a modified version of [GraphLab
PowerGraph](https://github.com/dato-code/PowerGraph).  Our code is
placed in
[apps/ms-ppr](https://github.com/lqhl/PowerGraph/tree/master/apps/ms-ppr).

Our implementation of offline indexing described in our paper is built on
[VENUS](https://lqhl.me/publication/icde2015.pdf) which is a disk-based graph
processing system developed by me with Huawei Noah's Ark Lab. Unfortunately,
VENUS cannot not be open sourced.  In this repository, we include an
implementation of our offline indexing algorithm atop PowerGraph.  However, the
offline indexing on PowerGraph is much slower than the version on VENUS.  To
solve this problem, I plan to provide a binary version, or migrate my program
to GraphChi in future.

Compiling
---------

For the detailed compiling process please refer to [PowerGraph's readme
file](README-PowerGraph.md). Here is a simple example:
```
./configure
cd release/apps/ms-ppr/
make -j4
```

Running
-------

Input file `edges.txt`:
```
1 2  # an edge from vertex 1 to 2
3 4  # an edge from vertex 3 to 4
```

Input file `sources.txt`:
```
3  # number of querying vertices
1
2
4
```

Online query without offline indexing:
```
./query-flow-v2 --graph /path/to/edges.txt \  # edge list
    --threshold 0.00001 --niters 8 \          # threshold and number of iterations
    --sources_file /path/to/sources.txt \     # querying vertices
    --num_sources 2 \                         # number of querying vertices (can be smaller than the one in the input file)
    --no_index true                           # run without offline index
```

Offline indexing:
```
./rw-fullpath --graph /path/to/edges.txt \  # edge list
    --niters 10 \                           # number of iterations
    --R 1 \                                 # number of random walks from each vertex
    --bin_prefix /path/to/ppr.index         # path to output graph and index
```

Online query with offline index:
```
./query-flow-v2 --graph /path/to/ppr.index \  # path to graph and index
    --sources_file /path/to/sources.txt \     # querying vertices
    --threshold 0.00001 --niters 6 \          # threshold and number of iterations
    --num_sources 2                           # number of querying vertices (can be smaller than the one in the input file)
```
