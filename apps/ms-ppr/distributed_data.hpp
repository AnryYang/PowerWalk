#ifndef GRAPHLAB_DISTRIBUTED_DATA_HPP
#define GRAPHLAB_DISTRIBUTED_DATA_HPP

#ifndef __NO_OPENMP__
#include <omp.h>
#endif

#include <boost/concept/requires.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <graphlab/logger/logger.hpp>
#include <graphlab/logger/assertions.hpp>

#include <graphlab/rpc/dc_dist_object.hpp>
#include <graphlab/graph/graph_basic_types.hpp>

#include <graphlab/macros_def.hpp>
namespace graphlab {

    template<typename Data>
    class distributed_data {
    public:
        typedef Data data_type;
        BOOST_CONCEPT_ASSERT((boost::DefaultConstructible<Data>));
        BOOST_CONCEPT_ASSERT((graphlab::Serializable<Data>));

        typedef boost::unordered_map<graphlab::vertex_id_type, Data> vid2data_map_type;
        typedef boost::unordered_map<graphlab::vertex_id_type, simple_spinlock> lock_manager_type;

    private:
        // creates a local dc_dist_object context
        graphlab::dc_dist_object<distributed_data> rmi;

        vid2data_map_type local_data;
        lock_manager_type vlocks;

    public:
        distributed_data(distributed_control& dc,
                boost::unordered_set<graphlab::vertex_id_type>* sources):
            rmi(dc, this) {
            foreach(const graphlab::vertex_id_type& source, *sources) {
                local_data[source] = Data();
                vlocks[source] = simple_spinlock();
            }
            rmi.barrier();
        }

        data_type& get_data(graphlab::vertex_id_type v) {
            auto it = local_data.find(v);
            return it->second;
        }

        void lock(graphlab::vertex_id_type v) {
            auto it = vlocks.find(v);
            it->second.lock();
        }

        void unlock(graphlab::vertex_id_type v) {
            auto it = vlocks.find(v);
            it->second.unlock();
        }
    };
} // end of namespace graphlab
#include <graphlab/macros_undef.hpp>

#endif
