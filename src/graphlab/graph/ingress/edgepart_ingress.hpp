#ifndef GRAPHLAB_EDGEPART_INGRESS_HPP
#define GRAPHLAB_EDGEPART_INGRESS_HPP

#include <boost/functional/hash.hpp>

#include <graphlab/rpc/buffered_exchange.hpp>
#include <graphlab/graph/graph_basic_types.hpp>
#include <graphlab/graph/ingress/distributed_ingress_base.hpp>
#include <graphlab/graph/distributed_graph.hpp>


#include <graphlab/macros_def.hpp>
namespace graphlab {
  template<typename VertexData, typename EdgeData>
  class distributed_graph;

  /**
   * \brief Ingress object assigning edges using randoming hash function.
   */
  template<typename VertexData, typename EdgeData>
  class edgepart_ingress : 
    public distributed_ingress_base<VertexData, EdgeData> {
  public:
    typedef distributed_graph<VertexData, EdgeData> graph_type;
    /// The type of the vertex data stored in the graph 
    typedef VertexData vertex_data_type;
    /// The type of the edge data stored in the graph 
    typedef EdgeData   edge_data_type;


    typedef distributed_ingress_base<VertexData, EdgeData> base_type;
   
  public:
    edgepart_ingress(distributed_control& dc, graph_type& graph) :
    base_type(dc, graph) {
    } // end of constructor

    ~edgepart_ingress() { }

    /** Add an vertex to the ingress object. */
    void add_vertex(vertex_id_type vid, const VertexData& vdata, const
            procid_t& procid) {
        ASSERT_EQ(this->rpc.procid(), 0);
        if (vid >= this->masters.size())
            this->masters.resize(vid + 1);
        this->masters[vid] = procid;
    }

    /** Add an edge to the ingress object using random assignment. */
    void add_edge(vertex_id_type source, vertex_id_type target,
                  const EdgeData& edata, const procid_t& procid) {
      typedef typename base_type::edge_buffer_record edge_buffer_record;
      const edge_buffer_record record(source, target, edata);
#ifdef _OPENMP
      base_type::edge_exchange.send(procid, record, omp_get_thread_num());
#else      
      base_type::edge_exchange.send(procid, record);
#endif
    } // end of add edge
  }; // end of distributed_random_ingress
}; // end of namespace graphlab
#include <graphlab/macros_undef.hpp>


#endif
