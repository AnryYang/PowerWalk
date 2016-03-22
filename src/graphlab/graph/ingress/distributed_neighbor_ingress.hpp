/**
 * Copyright (c) 2016 Qin Liu
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 */

#ifndef GRAPHLAB_DISTRIBUTED_NEIGHBOR_INGRESS_HPP
#define GRAPHLAB_DISTRIBUTED_NEIGHBOR_INGRESS_HPP


#include <graphlab/graph/graph_basic_types.hpp>
#include <graphlab/graph/ingress/distributed_ingress_base.hpp>
#include <graphlab/graph/ingress/ingress_edge_decision.hpp>
#include <graphlab/graph/distributed_graph.hpp>
#include <graphlab/rpc/buffered_exchange.hpp>
#include <graphlab/rpc/distributed_event_log.hpp>
#include <graphlab/parallel/pthread_tools.hpp>
#include <graphlab/macros_def.hpp>
namespace graphlab {
template<typename VertexData, typename EdgeData>
    class distributed_graph;

void vid_max_equal(vertex_id_type& a, const vertex_id_type& b) {
    a = std::max(a, b);
}

template<typename ValueType, typename KeyType, typename IdxType = vertex_id_type>
class MinHeap {
private:
    IdxType n;
    std::vector<std::pair<ValueType, KeyType>> heap;
    std::vector<IdxType> key2idx;

public:
    MinHeap() : n(0), heap(), key2idx() { }

    IdxType shift_up(IdxType cur) {
        if (cur == 0) return 0;
        IdxType p = (cur-1) / 2;

        if (heap[cur].first < heap[p].first) {
            std::swap(heap[cur], heap[p]);
            std::swap(key2idx[heap[cur].second], key2idx[heap[p].second]);
            return shift_up(p);
        }
        return cur;
    }

    void shift_down(IdxType cur) {
        IdxType l = cur*2 + 1;
        IdxType r = cur*2 + 2;

        if (l >= n)
            return;

        IdxType m = cur;
        if (heap[l].first < heap[cur].first)
            m = l;
        if (r < n && heap[r].first < heap[m].first)
            m = r;

        if (m != cur) {
            std::swap(heap[cur], heap[m]);
            std::swap(key2idx[heap[cur].second], key2idx[heap[m].second]);
            shift_down(m);
        }
    }

    void insert(ValueType value, KeyType key) {
        heap[n] = std::make_pair(value, key);
        key2idx[key] = n++;
        IdxType cur = shift_up(n-1);
        shift_down(cur);
    }

    bool contains(KeyType key) {
        return key2idx[key] != (IdxType) -1;
    }

    void decrease_key(KeyType key) {
        IdxType cur = key2idx[key];
        ASSERT_NE(cur, (IdxType) -1);
        ASSERT_NE(heap[cur].first, 0);

        if (heap[cur].first > 1) {
            heap[cur].first--;
            shift_up(cur);
        } else {
            std::swap(heap[cur], heap[n-1]);
            std::swap(key2idx[heap[cur].second], key2idx[key]);
            key2idx[key] = -1;
            n--;
            cur = shift_up(cur);
            shift_down(cur);
        }
    }

    void remove(KeyType key) {
        IdxType cur = key2idx[key];
        if (cur == (IdxType) -1)
            return;

        std::swap(heap[cur], heap[n-1]);
        std::swap(key2idx[heap[cur].second], key2idx[key]);
        key2idx[key] = -1;
        n--;
        cur = shift_up(cur);
        shift_down(cur);
    }

    bool get_min(ValueType& value, KeyType& key) {
        if (n > 0) {
            value = heap[0].first;
            key = heap[0].second;
            return true;
        } else
            return false;
    }

    void reset(KeyType max_key) {
        n = 0;
        heap.resize(max_key);
        key2idx.assign(max_key, -1);
    }

    void clear() {
        n = 0;
        heap.clear();
        key2idx.clear();
    }
};

/**
 * \brief Ingress object assigning edges using randoming hash function.
 */
template<typename VertexData, typename EdgeData>
    class distributed_neighbor_ingress:
        public distributed_ingress_base<VertexData, EdgeData> {
        public:
            typedef distributed_graph<VertexData, EdgeData> graph_type;
            /// The type of the vertex data stored in the graph
            typedef VertexData vertex_data_type;
            /// The type of the edge data stored in the graph
            typedef EdgeData   edge_data_type;

            typedef typename graph_type::vertex_record vertex_record;
            typedef typename graph_type::mirror_type mirror_type;

            typedef distributed_ingress_base<VertexData, EdgeData> base_type;

            dc_dist_object<distributed_neighbor_ingress> rmi;

            // Edge buffers used to build the temporary graph
            struct edge_record {
                vertex_id_type source, target;
                edge_data_type edata;
                bool reverse;
                edge_record(const vertex_id_type& source = vertex_id_type(-1),
                            const vertex_id_type& target = vertex_id_type(-1),
                            const edge_data_type& edata = edge_data_type(),
                            const bool reverse = false) :
                    source(source), target(target), edata(edata), reverse(reverse) { }
                void load(iarchive& arc) { arc >> source >> target >> edata >> reverse; }
                void save(oarchive& arc) const { arc << source << target << edata << reverse; }
            };
            buffered_exchange<edge_record> edge_record_exchange;

            typedef typename buffered_exchange<edge_record>::buffer_type edge_record_buffer_type;

            struct edge_record_comparator {
                bool operator()(const edge_record& a, const edge_record& b) {
                    if (a.source == b.source)
                        return a.target < b.target;
                    else
                        return a.source < b.source;
                }
            };

            struct candidate_type {
                vertex_id_type vid;
                std::vector<vertex_id_type> neighbors;
                candidate_type(const vertex_id_type& vid = vertex_id_type(-1)) :
                    vid(vid), neighbors() { }
                void load(iarchive& arc) { arc >> vid >> neighbors; }
                void save(oarchive& arc) const { arc << vid << neighbors; }
            };

            std::vector<edge_record> local_edges;
            std::vector<vertex_id_type> local_degrees;
            std::vector<size_t> start_ptr;
            vertex_id_type local_nvertices;
            size_t total_edges, nedges_limit, avg_degree;
            vertex_id_type max_vid;

            dense_bitset is_core, is_boundary;
            MinHeap<vertex_id_type, vertex_id_type, vertex_id_type> min_heap;

        public:
            distributed_neighbor_ingress(distributed_control& dc, graph_type& graph) :
                    base_type(dc, graph), rmi(dc, this),
#ifdef _OPENMP
                    edge_record_exchange(dc, omp_get_max_threads()) {
#else
                    edge_record_exchange(dc) {
#endif
            }

            ~distributed_neighbor_ingress() { }

            /** Add an edge to the ingress object using neighbor greedy assignment. */
            void add_edge(vertex_id_type source, vertex_id_type target,
                    const EdgeData& edata) {
                edge_record record(source, target, edata, false),
                    reversed(target, source, edata, true);
#ifdef _OPENMP
                edge_record_exchange.send(source % rmi.numprocs(), record, omp_get_thread_num());
                edge_record_exchange.send(target % rmi.numprocs(), reversed, omp_get_thread_num());
#else
                edge_record_exchange.send(source % rmi.numprocs(), record);
                edge_record_exchange.send(target % rmi.numprocs(), reversed);
#endif
            } // end of add edge

            virtual void finalize() {
                edge_record_exchange.flush();
                if (rmi.procid() == 0)
                    logstream(LOG_INFO) << "Flushed edge records..." << std::endl;

                procid_t proc;
                edge_record_buffer_type edge_record_buffer;
                while (edge_record_exchange.recv(proc, edge_record_buffer)) {
                    local_edges.insert(local_edges.end(), edge_record_buffer.begin(), edge_record_buffer.end());
                }
                edge_record_exchange.clear();

                total_edges = local_edges.size();
                rmi.all_reduce(total_edges);
                if (total_edges == 0) {
                    logstream(LOG_INFO) << "Skipping Graph Finalization because no changes happened..." << std::endl;
                    return;
                }
                total_edges /= 2;
                nedges_limit = total_edges / rmi.numprocs();

                logstream(LOG_INFO) << "Proc " << rmi.procid() << ": " <<
                    "Number of local edges: " << local_edges.size() << std::endl;

                std::sort(local_edges.begin(), local_edges.end(), edge_record_comparator());
                local_nvertices = local_edges.empty() ? 0 :
                    (local_edges.back().source / rmi.numprocs() + 1);
                local_degrees.assign(local_nvertices, 0);
                for (auto& each : local_edges)
                    local_degrees[each.source / rmi.numprocs()]++;
                start_ptr.assign(local_nvertices, 0);
                for (vertex_id_type i = 1; i < local_nvertices; i++)
                    start_ptr[i] = start_ptr[i-1] + local_degrees[i-1];

                if (local_edges.empty())
                    max_vid = 0;
                else
                    max_vid = local_edges.back().source;
                rmi.all_reduce2(max_vid, vid_max_equal);

                avg_degree = total_edges * 2 / max_vid;

                if (rmi.procid() == 0) {
                    logstream(LOG_INFO) << "Max vertex ID: " << max_vid << std::endl;
                    logstream(LOG_INFO) << "Expected edges on each machine: " << nedges_limit << std::endl;
                    logstream(LOG_INFO) << "Average degree: " << avg_degree << std::endl;
                }

                is_core.resize(max_vid);
                is_boundary.resize(max_vid);

                for (procid_t p = 0; p < rmi.numprocs()-1; p++) {
                    min_heap.reset(local_nvertices);
                    is_core.clear();
                    is_boundary.clear();

                    if (p == rmi.procid()) {
                        logstream(LOG_INFO) << "Start master " << p <<
                            std::endl;
                        master();
                    } else
                        slave(p);
                    rmi.barrier();
                }

                if (rmi.procid() == 0)
                    logstream(LOG_INFO) << "Flush remaining edges..." << std::endl;
                size_t num_allocated_edges = 0;
                for (vertex_id_type i = 0; i < local_nvertices; i++)
                    for (size_t ptr = start_ptr[i]; ptr < start_ptr[i] + local_degrees[i]; ptr++)
                        if (!local_edges[ptr].reverse) {
                            num_allocated_edges++;
                            typename base_type::edge_buffer_record record(local_edges[ptr].source, local_edges[ptr].target, local_edges[ptr].edata);
#ifdef _OPENMP
                            base_type::edge_exchange.send(rmi.numprocs()-1, record, omp_get_thread_num());
#else
                            base_type::edge_exchange.send(rmi.numprocs()-1, record);
#endif
                        }
                rmi.all_reduce(num_allocated_edges);
                if (rmi.procid() == rmi.numprocs()-1)
                    logstream(LOG_INFO) << "Allocated edges: " << num_allocated_edges << std::endl;

                local_edges.clear();
                local_degrees.clear();
                start_ptr.clear();
                min_heap.clear();

                distributed_ingress_base<VertexData, EdgeData>::finalize();
            }

            void master() {
                size_t num_allocated_edges = 0;

                while (true) {
                    if (num_allocated_edges >= nedges_limit)
                        break;

                    std::vector<request_future<candidate_type>> futures;
                    for (procid_t procid = 0; procid < rmi.numprocs(); procid++)
                        futures.push_back(rmi.future_remote_request(procid,
                                    &distributed_neighbor_ingress::get_candidate));

                    std::vector<candidate_type> candidates;
                    size_t avg = 0;
                    for (auto& each : futures) {
                        candidate_type c = each();
                        if (c.vid != (vertex_id_type) -1) {
                            candidates.push_back(c);
                            avg += c.neighbors.size();
                        }
                    }
                    if (candidates.size() > 0)
                        avg = (avg+1) / candidates.size();
                    avg = std::min(avg_degree, avg);

                    std::vector<vertex_id_type> new_cores;
                    boost::unordered_set<vertex_id_type> neighbors;
                    for (candidate_type& c : candidates)
                        if (c.neighbors.size() <= avg) {
                            is_core.set_bit(c.vid);
                            new_cores.push_back(c.vid);
                            for (auto& u : c.neighbors) {
                                is_boundary.set_bit(u);
                                neighbors.insert(u);
                            }
                        }

                    rmi.broadcast(new_cores, true);
                    rmi.broadcast(neighbors, true);

                    if (new_cores.empty())
                        break;

                    size_t count = update(rmi.procid(), new_cores, neighbors);
                    rmi.all_reduce(count);
                    num_allocated_edges += count;
                    if (num_allocated_edges >= nedges_limit) {
                        logstream(LOG_INFO) << "#avg: " << avg << std::endl;
                        logstream(LOG_INFO) << "#new edges: " << count << ", #neighbors: " << neighbors.size() << std::endl;
                    }
                }
                logstream(LOG_INFO) << "Allocated edges: " << num_allocated_edges << std::endl;
            }

            void slave(procid_t master_id) {
                size_t num_allocated_edges = 0;

                while (true) {
                    if (num_allocated_edges >= nedges_limit)
                        break;

                    std::vector<vertex_id_type> new_cores;
                    boost::unordered_set<vertex_id_type> neighbors;
                    rmi.broadcast(new_cores, false);
                    rmi.broadcast(neighbors, false);

                    if (new_cores.empty())
                        break;

                    for (auto& u : new_cores)
                        is_core.set_bit(u);
                    for (auto& u : neighbors)
                        is_boundary.set_bit(u);

                    size_t count = update(master_id, new_cores, neighbors);
                    rmi.all_reduce(count);
                    num_allocated_edges += count;
                }
            }

            candidate_type get_candidate() {
                candidate_type candidate;
                vertex_id_type degree, vid;
                if (local_nvertices == 0)
                    return candidate;
                if (!min_heap.get_min(degree, vid)) {
                    int count = 0;
                    do {
                        vid = random::fast_uniform(vertex_id_type(0), local_nvertices-1);
                        if (count++ >= 10)
                            return candidate;
                    } while (local_degrees[vid] == 0 || local_degrees[vid] >= avg_degree);
                } else {
                    ASSERT_EQ(degree, local_degrees[vid]);
                }
                candidate.vid = vid * rmi.numprocs() + rmi.procid();
                ASSERT_EQ(candidate.vid, local_edges[start_ptr[vid]].source);
                for (size_t ptr = start_ptr[vid]; ptr < start_ptr[vid] + local_degrees[vid]; ptr++) {
                    candidate.neighbors.push_back(local_edges[ptr].target);
                }
                return candidate;
            }

            size_t update(procid_t master_id, std::vector<vertex_id_type>& new_cores, boost::unordered_set<vertex_id_type>& neighbors) {
                size_t count = 0;

                for (size_t i = 0; i < new_cores.size(); i++) {
                    vertex_id_type u = new_cores[i];
                    if (u % rmi.numprocs() == rmi.procid()) {
                        u /= rmi.numprocs();
                        ASSERT_LT(u, local_nvertices);
                        ASSERT_GT(local_degrees[u], 0);
                        for (size_t ptr = start_ptr[u]; ptr < start_ptr[u] + local_degrees[u]; ptr++)
                            if (!local_edges[ptr].reverse) {
                                count++;
                                typename base_type::edge_buffer_record record(local_edges[ptr].source, local_edges[ptr].target, local_edges[ptr].edata);
#ifdef _OPENMP
                                base_type::edge_exchange.send(master_id, record, omp_get_thread_num());
#else
                                base_type::edge_exchange.send(master_id, record);
#endif
                            }
                        local_degrees[u] = 0;
                        min_heap.remove(u);
                    }
                }

                for (auto u : neighbors) {
                    if (u % rmi.numprocs() == rmi.procid()) {
                        u /= rmi.numprocs();
                        ASSERT_LT(u, local_nvertices);
                        if (local_degrees[u] > 0) {
                            if (!min_heap.contains(u))
                                min_heap.insert(local_degrees[u], u);
                            for (size_t ptr = start_ptr[u]; ptr < start_ptr[u] + local_degrees[u]; )
                                if (is_core.get(local_edges[ptr].target) || is_boundary.get(local_edges[ptr].target)) {
                                    if (!local_edges[ptr].reverse) {
                                        count++;
                                        typename base_type::edge_buffer_record record(local_edges[ptr].source, local_edges[ptr].target, local_edges[ptr].edata);
#ifdef _OPENMP
                                        base_type::edge_exchange.send(master_id, record, omp_get_thread_num());
#else
                                        base_type::edge_exchange.send(master_id, record);
#endif
                                    }
                                    local_degrees[u]--;
                                    if (local_degrees[u] > 0)
                                        std::swap(local_edges[ptr], local_edges[start_ptr[u] + local_degrees[u]]);
                                    min_heap.decrease_key(u);
                                } else
                                    ptr++;
                        }
                    }
                }
                return count;
            }

        }; // end of distributed_neighbor_ingress

}; // end of namespace graphlab
#include <graphlab/macros_undef.hpp>


#endif
