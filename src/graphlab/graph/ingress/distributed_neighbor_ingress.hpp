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


#include <unordered_map>

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
    std::unordered_map<KeyType, IdxType> key2idx;

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
        return key2idx.count(key);
    }

    void decrease_key(KeyType key) {
        auto it = key2idx.find(key);
        ASSERT_TRUE(it != key2idx.end());
        IdxType cur = it->second;
        ASSERT_NE(heap[cur].first, 0);

        if (heap[cur].first > 1) {
            heap[cur].first--;
            shift_up(cur);
        } else {
            n--;
            if (n > 0) {
                heap[cur] = heap[n];
                key2idx[heap[cur].second] = cur;
                cur = shift_up(cur);
                shift_down(cur);
            }
            key2idx.erase(it);
        }
    }

    bool remove(KeyType key) {
        auto it = key2idx.find(key);
        if (it == key2idx.end())
            return false;
        IdxType cur = it->second;

        n--;
        if (n > 0) {
            heap[cur] = heap[n];
            key2idx[heap[cur].second] = cur;
            cur = shift_up(cur);
            shift_down(cur);
        }
        key2idx.erase(it);
        return true;
    }

    bool get_min(ValueType& value, KeyType& key) {
        if (n > 0) {
            value = heap[0].first;
            key = heap[0].second;
            return true;
        } else
            return false;
    }

    void reset(IdxType nelements) {
        n = 0;
        heap.resize(nelements);
        key2idx.clear();
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

            struct out_edge {
                edge_data_type edata;
                bool reverse;
                out_edge(const edge_data_type& edata = edge_data_type(), const
                        bool reverse = false) : edata(edata), reverse(reverse)
                {
                }
            };

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

            struct candidate_comparator {
                bool operator()(const candidate_type& a, const candidate_type& b) {
                    return a.neighbors.size() < b.neighbors.size();
                }
            };

            size_t total_edges, nedges_limit;
            double avg_degree;
            std::map<vertex_id_type, std::multimap<vertex_id_type, out_edge>> adj_out;
            std::map<vertex_id_type, std::set<vertex_id_type>> adj_in;

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
                adj_out.clear();
                adj_in.clear();
                total_edges = 0;
                vertex_id_type max_vid = 0;
                while (edge_record_exchange.recv(proc, edge_record_buffer)) {
                    for (auto& e : edge_record_buffer) {
                        total_edges++;
                        max_vid = std::max(max_vid, e.source);
                        adj_out[e.source].emplace(e.target, out_edge(e.edata, e.reverse));
                        adj_in[e.target].insert(e.source);
                    }
                }
                max_vid++;
                edge_record_exchange.clear();

                logstream(LOG_INFO) << "Proc " << rmi.procid() << ": " <<
                    "Number of local edges: " << total_edges << std::endl;

                rmi.all_reduce(total_edges);
                if (total_edges == 0) {
                    logstream(LOG_INFO) << "Skipping Graph Finalization because no changes happened..." << std::endl;
                    return;
                }
                total_edges /= 2;
                nedges_limit = total_edges / rmi.numprocs();

                vertex_id_type nvertices = adj_out.size();
                rmi.all_reduce(nvertices);
                avg_degree = (double) total_edges * 2 / nvertices;

                rmi.all_reduce2(max_vid, vid_max_equal);

                if (rmi.procid() == 0) {
                    logstream(LOG_INFO) << "Max vertex ID: " << max_vid << std::endl;
                    logstream(LOG_INFO) << "Number of vertices: " << nvertices << std::endl;
                    logstream(LOG_INFO) << "Number of edges: " << total_edges << std::endl;
                    logstream(LOG_INFO) << "Expected edges on each machine: " << nedges_limit << std::endl;
                    logstream(LOG_INFO) << "Average degree: " << avg_degree << std::endl;
                }

                is_core.resize(max_vid);
                is_boundary.resize(max_vid);

                for (procid_t p = 0; p < rmi.numprocs()-1; p++) {
                    min_heap.reset(adj_out.size());
                    is_core.clear();
                    is_boundary.clear();

                    if (p == rmi.procid()) {
                        logstream(LOG_INFO) << "Start master " << p <<
                            std::endl;
                        master();
                    } else
                        slave(p);
                }

                if (rmi.procid() == 0)
                    logstream(LOG_INFO) << "Flush remaining edges..." << std::endl;
                size_t num_allocated_edges = 0;
                for (auto& u_adj : adj_out)
                    for (auto e : u_adj.second)
                        if (!e.second.reverse) {
                            num_allocated_edges++;
                            typename base_type::edge_buffer_record record(u_adj.first, e.first, e.second.edata);
#ifdef _OPENMP
                            base_type::edge_exchange.send(rmi.numprocs()-1, record, omp_get_thread_num());
#else
                            base_type::edge_exchange.send(rmi.numprocs()-1, record);
#endif
                        }
                rmi.all_reduce(num_allocated_edges);
                if (rmi.procid() == rmi.numprocs()-1)
                    logstream(LOG_INFO) << "Allocated edges: " << num_allocated_edges << std::endl;

                adj_out.clear();
                adj_in.clear();
                min_heap.clear();

                distributed_ingress_base<VertexData, EdgeData>::finalize();
            }

            void master() {
                size_t num_allocated_edges = 0;

                int iteration = 0;
                while (num_allocated_edges < nedges_limit) {
                    rmi.barrier();
                    iteration++;
                    std::vector<request_future<std::vector<candidate_type>>> futures;
                    for (procid_t procid = 0; procid < rmi.numprocs(); procid++)
                        if (iteration < 10)
                            futures.push_back(rmi.future_remote_request(procid,
                                        &distributed_neighbor_ingress::get_candidate, 1));
                        else
                            futures.push_back(rmi.future_remote_request(procid,
                                        &distributed_neighbor_ingress::get_candidate, 10));

                    std::vector<candidate_type> candidates;
                    size_t avg = 0;
                    for (auto& each : futures)
                        for (auto& c : each()) {
                            candidates.push_back(c);
                            avg += c.neighbors.size();
                        }
                    if (candidates.size() > 0)
                        avg = (avg+1) / candidates.size();

                    if (candidates.empty() && !adj_out.empty()) {
                        int count = 0;
                        auto it = adj_out.begin();
                        std::advance(it, random::rand() % adj_out.size());
                        while (it->second.empty() || it->second.size() > avg_degree) {
                            it++;
                            if (count++ >= adj_out.size())
                                break;
                        }
                        if (count < adj_out.size()) {
                            candidate_type c;
                            c.vid = it->first;
                            for (auto u : it->second) {
                                c.neighbors.push_back(u.first);
                            }
                            candidates.push_back(c);
                            avg = c.neighbors.size();
                        }
                    }

                    std::sort(candidates.begin(), candidates.end(), candidate_comparator());

                    std::vector<vertex_id_type> new_cores;
                    boost::unordered_set<vertex_id_type> neighbors;
                    for (candidate_type& c : candidates) {
                        if (neighbors.size() >= 100)
                            break;
                        if (c.neighbors.size() <= avg) {
                            is_core.set_bit(c.vid);
                            new_cores.push_back(c.vid);
                            for (auto& u : c.neighbors) {
                                is_boundary.set_bit(u);
                                neighbors.insert(u);
                            }
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
                logstream(LOG_INFO) << "#iterations: " << iteration << std::endl;
                logstream(LOG_INFO) << "Allocated edges: " << num_allocated_edges << std::endl;
            }

            void slave(procid_t master_id) {
                size_t num_allocated_edges = 0;

                while (num_allocated_edges < nedges_limit) {
                    rmi.barrier();
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

            std::vector<candidate_type> get_candidate(int c) {
                std::vector<candidate_type> res;
                candidate_type candidate;
                vertex_id_type degree, vid;
                while (c-- > 0 && min_heap.get_min(degree, vid)) {
                    ASSERT_EQ(degree, adj_out[vid].size());
                    candidate.vid = vid;
                    candidate.neighbors.clear();
                    for (auto& e : adj_out[vid]) {
                        candidate.neighbors.push_back(e.first);
                    }
                    res.push_back(candidate);
                    ASSERT_TRUE(min_heap.remove(vid));
                }
                for (auto& c : res)
                    min_heap.insert(c.neighbors.size(), c.vid);
                return res;
            }

            size_t update(procid_t master_id, std::vector<vertex_id_type>& new_cores, boost::unordered_set<vertex_id_type>& neighbors) {
                size_t count = 0;

                for (size_t i = 0; i < new_cores.size(); i++) {
                    vertex_id_type u = new_cores[i];
                    if (u % rmi.numprocs() == rmi.procid()) {
                        ASSERT_GT(adj_out.count(u), 0);
                        for (auto& e : adj_out[u]) {
                            if (!e.second.reverse) {
                                count++;
                                typename base_type::edge_buffer_record record(u, e.first, e.second.edata);
#ifdef _OPENMP
                                base_type::edge_exchange.send(master_id, record, omp_get_thread_num());
#else
                                base_type::edge_exchange.send(master_id, record);
#endif
                            }
                            adj_in[e.first].erase(u);
                        }
                        adj_out.erase(u);
                        min_heap.remove(u);
                    }
                }

                for (auto u : neighbors) {
                    if (u % rmi.numprocs() == rmi.procid()) {
                        auto it = adj_out.find(u);
                        if (it != adj_out.end() && !it->second.empty()) {
                            if (!min_heap.contains(u))
                                min_heap.insert(it->second.size(), u);
                            for (auto e = it->second.begin(); e != it->second.end(); )
                                if (is_core.get(e->first) || is_boundary.get(e->first)) {
                                    if (!e->second.reverse) {
                                        count++;
                                        typename base_type::edge_buffer_record record(u, e->first, e->second.edata);
#ifdef _OPENMP
                                        base_type::edge_exchange.send(master_id, record, omp_get_thread_num());
#else
                                        base_type::edge_exchange.send(master_id, record);
#endif
                                    }
                                    min_heap.decrease_key(u);
                                    adj_in[e->first].erase(u);
                                    e = it->second.erase(e);
                                } else
                                    e++;
                        }
                    }
                }

                for (auto u : neighbors) {
                    auto it = adj_in.find(u);
                    if (it == adj_in.end()) continue;
                    for (auto e = it->second.begin(); e != it->second.end(); )
                        if (is_boundary.get(*e)) {
                            auto it2 = adj_out[*e].find(u);
                            ASSERT_TRUE(it2 != adj_out[*e].end());
                            while (it2 != adj_out[*e].end()) {
                                if (!it2->second.reverse) {
                                    count++;
                                    typename base_type::edge_buffer_record record(*e, u, it2->second.edata);
#ifdef _OPENMP
                                    base_type::edge_exchange.send(master_id, record, omp_get_thread_num());
#else
                                    base_type::edge_exchange.send(master_id, record);
#endif
                                }
                                min_heap.decrease_key(*e);
                                adj_out[*e].erase(it2);
                                it2 = adj_out[*e].find(u);
                            }
                            e = it->second.erase(e);
                        } else
                            e++;
                }

                return count;
            }

        }; // end of distributed_neighbor_ingress

}; // end of namespace graphlab
#include <graphlab/macros_undef.hpp>


#endif
