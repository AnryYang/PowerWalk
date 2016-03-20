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
                    return a.source < b.source ? true : a.target < b.target;
                }
            };

            size_t total_edges, nedges_limit;
            vertex_id_type max_vid;

        public:
            distributed_neighbor_ingress(distributed_control& dc, graph_type& graph) :
                    base_type(dc, graph),
#ifdef _OPENMP
                    edge_record_exchange(dc, omp_get_max_threads()) {
#else
                    edge_record_exchange(dc) {
#endif
                // TODO: initialization
            }

            ~distributed_neighbor_ingress() { }

            /** Add an edge to the ingress object using neighbor greedy assignment. */
            void add_edge(vertex_id_type source, vertex_id_type target,
                    const EdgeData& edata) {
                // TODO: build a temporary graph in distirbuted memeory.
                // each edge is stored twice (need to mark the direction).
                edge_record record(source, target, edata, false),
                    reversed(target, source, edata, true);
#ifdef _OPENMP
                edge_record_exchange.send(source % base_type::rpc.numprocs(), record, omp_get_thread_num());
                edge_record_exchange.send(target % base_type::rpc.numprocs(), reversed, omp_get_thread_num());
#else
                edge_record_exchange.send(source % base_type::rpc.numprocs(), record);
                edge_record_exchange.send(target % base_type::rpc.numprocs(), reversed);
#endif
            } // end of add edge

            virtual void finalize() {
                // TODO 1: build the temporary graph (need to use a bool to
                // mark deletions).
                edge_record_exchange.flush();
                if (base_type::rpc.procid() == 0) {
                    logstream(LOG_INFO) << "Flushed edge records..." << std::endl;
                }

                procid_t proc;
                edge_record_buffer_type edge_record_buffer;
                std::vector<edge_record> local_edges;
                while (edge_record_exchange.recv(proc, edge_record_buffer)) {
                    local_edges.insert(local_edges.end(), edge_record_buffer.begin(), edge_record_buffer.end());
                }
                edge_record_exchange.clear();

                total_edges = edge_record_exchange.size();
                base_type::rpc.all_reduce(total_edges);
                total_edges /= 2;
                nedges_limit = total_edges / base_type::rpc.numprocs();

                std::sort(local_edges.begin(), local_edges.end(), edge_record_comparator());
                vertex_id_type local_nvertices = local_edges.empty() ? 0 :
                    (local_edges.back().source / base_type::rpc.numprocs() + 1);
                std::vector<vertex_id_type> local_degrees(local_nvertices, 0);
                for (auto& each : local_edges)
                    local_degrees[each.source / base_type::rpc.numprocs()]++;
                std::vector<size_t> start_ptr(local_nvertices, 0);
                for (vertex_id_type i = 1; i < local_nvertices; i++)
                    start_ptr[i] = start_ptr[i-1] + local_degrees[i-1];

                max_vid = local_nvertices;
                base_type::rpc.all_reduce2(max_vid, vid_max_equal);

                // TODO 2: main algorithm
                // for i = 0 to P-1:
                //   if i == P-1:
                //     send all remaining edges to machine P-1
                //   else if procid == i:
                //     master()
                //   else:
                //     slave()
                for (procid_t i = 0; i < base_type::rpc.numprocs(); i++)
                    if (i == base_type::rpc.numprocs()-1) {
                        for (vertex_id_type i = 0; i < local_nvertices; i++)
                            for (size_t ptr = start_ptr[i]; ptr < start_ptr[i] + local_degrees[i]; ptr++) {
                                typedef typename base_type::edge_buffer_record edge_buffer_record;
                                edge_buffer_record record(local_edges[ptr].source, local_edges[ptr].target, local_edges[ptr].edata);
                                if (local_edges[ptr].reverse)
                                    std::swap(record.source, record.target);
#ifdef _OPENMP
                                base_type::edge_exchange.send(i, record, omp_get_thread_num());
#else      
                                base_type::edge_exchange.send(i, record);
#endif
                            }
                    } else if (i == base_type::rpc.procid())
                        master();
                    else
                        slave(i);

                distributed_ingress_base<VertexData, EdgeData>::finalize();
            }

            void master() {
                // TODO:
                //   while true:
                //     broadcast num_allocated_edges
                //     if num_allocated_edges > limit
                //       break
                //
                //     for each machine p
                //       remote_request(get_candidate_and_neighbors)
                //     barrier()
                //
                //     get cores from candidates // can we use as more candidates
                //     as possible?
                //
                //     broadcast new cores and boundaries
                //     update heap and send edges
                size_t num_allocalted_edges = 0;
                dense_bitset is_core(max_vid), is_boundary(max_vid);

                while (true) {
                    base_type::rpc.broadcast(num_allocalted_edges, true);
                    if (num_allocalted_edges >= nedges_limit)
                        break;
                }
            }

            void slave(procid_t master_id) {
                // TODO:
                //   while ture:
                //     broadcast num_allocated_edges
                //     if num_allocated_edges > limit
                //       break
                //
                //     barrier()
                //
                //     broadcast new cores and boundaries
                //     update heap and send edges
                size_t num_allocalted_edges = 0;
                dense_bitset is_core(max_vid), is_boundary(max_vid);

                while (true) {
                    base_type::rpc.broadcast(num_allocalted_edges, false);
                    if (num_allocalted_edges >= nedges_limit)
                        break;
                }
            }

            // get_candidate_and_neighbors():
            //   if heap.empty():
            //     return random vertex in this machine
            //   else:
            //     return min

        }; // end of distributed_neighbor_ingress

}; // end of namespace graphlab
#include <graphlab/macros_undef.hpp>


#endif
