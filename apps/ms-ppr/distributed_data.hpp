/*
 * Copyright (c) 2015 Qin Liu.
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
    template<typename Map, typename PlusEqual>
    struct map_plusequal {
        PlusEqual plusequal;
        map_plusequal(PlusEqual plusequal) : plusequal(plusequal) { }
        void operator()(Map& a, const Map& b) {
            for (auto it = b.begin(); it != b.end(); ++it) {
                plusequal(a[it->first], it->second);
            }
        }
    };

    template<typename Data>
    class distributed_data {
    public:
        typedef Data data_type;
        BOOST_CONCEPT_ASSERT((boost::DefaultConstructible<Data>));
        BOOST_CONCEPT_ASSERT((graphlab::Serializable<Data>));

        typedef boost::unordered_map<graphlab::vertex_id_type, Data> vid2data_map_type;
        typedef boost::unordered_map<graphlab::vertex_id_type, simple_spinlock> lock_manager_type;
        typedef void (*PlusEqual) (data_type& a, const data_type& b);

    private:
        // creates a local dc_dist_object context
        graphlab::dc_dist_object<distributed_data> rmi;

        vid2data_map_type local_data;
        lock_manager_type vlocks;
        PlusEqual plusequal;

    public:
        distributed_data(distributed_control& dc,
                boost::unordered_set<graphlab::vertex_id_type>* sources,
                PlusEqual plusequal):
            rmi(dc, this), plusequal(plusequal) {
            for (auto const& source: *sources) {
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

        void synchronize() {
            map_plusequal<vid2data_map_type, PlusEqual> func(plusequal);
            rmi.all_reduce2(local_data, func);
        }

        void add(graphlab::vertex_id_type v, data_type data) {
            vlocks[v].lock();
            plusequal(local_data[v], data);
            vlocks[v].unlock();
        }

        void reduce2one() {
            if (rmi.procid() > 0) {
                for (auto const& kv: local_data) {
                    rmi.remote_call(0, &distributed_data::add, kv.first,
                            kv.second);
                }
            }
        }
    };
} // end of namespace graphlab
#include <graphlab/macros_undef.hpp>

#endif
