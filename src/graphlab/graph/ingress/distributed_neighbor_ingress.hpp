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

        public:
            distributed_neighbor_ingress(distributed_control& dc, graph_type& graph) :
                    base_type(dc, graph) { 
                // TODO: initialization
            }

            ~distributed_neighbor_ingress() { }

            /** Add an edge to the ingress object using neighbor greedy assignment. */
            void add_edge(vertex_id_type source, vertex_id_type target,
                    const EdgeData& edata) {
                // TODO: build a temporary graph in distirbuted memeory.
                // each edge is stored twice (need to mark the direction).
            } // end of add edge

            virtual void finalize() {
                // TODO 1: build the temporary graph (need to use a bool to
                // mark deletions).

                // TODO 2: main algorithm
                // for i = 0 to P-1:
                //   if i == P-1:
                //     send all remaining edges to machine P-1
                //   else if procid == i:
                //     master()
                //   else:
                //     slave()
                //
                // master():
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
                //
                // slave():
                //   while ture:
                //     broadcast num_allocated_edges
                //     if num_allocated_edges > limit
                //       break
                //
                //     barrier()
                //
                //     broadcast new cores and boundaries
                //     update heap and send edges
                //
                // get_candidate_and_neighbors():
                //   if heap.empty():
                //     return random vertex in this machine
                //   else:
                //     return min

                distributed_ingress_base<VertexData, EdgeData>::finalize(); 
            }

        }; // end of distributed_neighbor_ingress

}; // end of namespace graphlab
#include <graphlab/macros_undef.hpp>


#endif
