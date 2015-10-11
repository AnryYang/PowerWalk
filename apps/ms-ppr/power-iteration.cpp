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
// NOTE: Copied from toolkits/graph_analytics/pagerank.cpp
#include <vector>
#include <string>
#include <fstream>

#include <graphlab.hpp>
// #include <graphlab/macros_def.hpp>

// Global random reset probability
double RESET_PROB = 0.15;

size_t ITERATIONS = 10;

graphlab::vertex_id_type source_vertex = 0;

// The vertex data is just the pagerank value (a double)
typedef double vertex_data_type;

// There is no edge data in the pagerank application
typedef graphlab::empty edge_data_type;

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;

/*
 * A simple function used by graph.transform_vertices(init_vertex);
 * to initialize the vertes data.
 */
void init_vertex(graph_type::vertex_type& vertex) {
    if (vertex.id() == source_vertex)
        vertex.data() = 1;
}

class pagerank : public graphlab::ivertex_program<graph_type, double, double> {
    private:
        double pr;
    public:
        void init(icontext_type& context, const vertex_type& vertex,
                const message_type& msg) {
            pr = msg;
        }

        edge_dir_type gather_edges(icontext_type& context,
                const vertex_type& vertex) const {
            return graphlab::IN_EDGES;
        }

        double gather(icontext_type& context, const vertex_type& vertex,
                edge_type& edge) const {
            return (edge.source().data() / edge.source().num_out_edges());
        }

        void apply(icontext_type& context, vertex_type& vertex,
                const gather_type& total) {
            double newval = (1.0 - RESET_PROB) * (pr + total);
            if (vertex.id() == source_vertex)
                newval += RESET_PROB;
            vertex.data() = newval;
            context.signal(vertex);
            if (vertex.num_out_edges() == 0)
                context.signal_vid(source_vertex, newval);
        }

        edge_dir_type scatter_edges(icontext_type& context,
                const vertex_type& vertex) const {
            return graphlab::NO_EDGES;
        }

        void scatter(icontext_type& context, const vertex_type& vertex,
                edge_type& edge) const {
        }

        void save(graphlab::oarchive& oarc) const {
        }

        void load(graphlab::iarchive& iarc) {
        }

    };


/*
 * We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", pagerank_writer()) to save the graph.
 */
struct pagerank_writer {
    std::string save_vertex(graph_type::vertex_type v) {
        std::stringstream strm;
        strm << v.id() << "\t" << v.data() << "\n";
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
}; // end of pagerank writer


double map_rank(const graph_type::vertex_type& v) {
    if (v.id() == source_vertex)
        return v.data();
    else
        return 0;
}

double pagerank_sum(graph_type::vertex_type v) {
    return v.data();
}

int main(int argc, char** argv) {
    // Initialize control plain using mpi
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    global_logger().set_log_level(LOG_INFO);

    // Parse command line options -----------------------------------------------
    graphlab::command_line_options clopts("PageRank algorithm.");
    std::string graph_dir;
    std::string format = "adj";
    clopts.attach_option("graph", graph_dir,
            "The graph file.  If none is provided "
            "then a toy graph will be created");
    clopts.add_positional("graph");
    clopts.attach_option("format", format,
            "The graph file format");
    size_t powerlaw = 0;
    clopts.attach_option("powerlaw", powerlaw,
            "Generate a synthetic powerlaw out-degree graph. ");
    clopts.attach_option("iterations", ITERATIONS,
            "Runs complete (non-dynamic) PageRank for a fixed "
            "number of iterations.");
    std::string saveprefix;
    clopts.attach_option("saveprefix", saveprefix,
            "If set, will save the resultant pagerank to a "
            "sequence of files with prefix saveprefix");
    clopts.attach_option("source_vertex", source_vertex,
            "The source vertex of the Personalized PageRank vector");

    if(!clopts.parse(argc, argv)) {
        dc.cout() << "Error in parsing command line arguments." << std::endl;
        return EXIT_FAILURE;
    }


    // make sure this is the synchronous engine
    dc.cout() << "running for " << ITERATIONS << " iterations." << std::endl;
    clopts.get_engine_args().set_option("max_iterations", ITERATIONS);

    // Build the graph ----------------------------------------------------------
    graph_type graph(dc, clopts);
    if(powerlaw > 0) { // make a synthetic graph
        dc.cout() << "Loading synthetic Powerlaw graph." << std::endl;
        graph.load_synthetic_powerlaw(powerlaw, false, 2.1, 100000000);
    }
    else if (graph_dir.length() > 0) { // Load the graph from a file
        dc.cout() << "Loading graph in format: "<< format << std::endl;
        graph.load_format(graph_dir, format);
    }
    else {
        dc.cout() << "graph or powerlaw option must be specified" << std::endl;
        clopts.print_description();
        return 0;
    }
    // must call finalize before querying the graph
    graph.finalize();
    dc.cout() << "#vertices: " << graph.num_vertices()
        << " #edges:" << graph.num_edges() << std::endl;

    // Initialize the vertex data
    graph.transform_vertices(init_vertex);

    // Running The Engine -------------------------------------------------------
    graphlab::synchronous_engine<pagerank> engine(dc, graph, clopts);
    engine.signal_all();
    engine.start();
    const double runtime = engine.elapsed_seconds();
    dc.cout() << "Finished Running engine in " << runtime
        << " seconds." << std::endl;

    const double source_rank = graph.map_reduce_vertices<double>(map_rank);
    dc.cout() << "Source Rank: " << source_rank << std::endl;

    const double total_rank = graph.map_reduce_vertices<double>(pagerank_sum);
    dc.cout() << "Total Rank: " << total_rank << "\n";

    // Save the final graph -----------------------------------------------------
    if (saveprefix != "") {
        graph.save(saveprefix, pagerank_writer(),
                false,    // do not gzip
                true,     // save vertices
                false);   // do not save edges
    }

    // Tear-down communication layer and quit -----------------------------------
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}
