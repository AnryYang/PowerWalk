#include <vector>
#include <string>
#include <fstream>
#include <map>

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <graphlab.hpp>

// Global random reset probability
double RESET_PROB = 0.15;
uint16_t num_walkers = 1000;
size_t niters;
int phase = 0;
boost::unordered_set<graphlab::vertex_id_type> *sources = NULL;

typedef boost::unordered_map<graphlab::vertex_id_type, uint16_t> map_t;

struct Counter {
    map_t counter;

    Counter() : counter() { }

    void save(graphlab::oarchive& oarc) const {
        oarc << counter;
    }

    void load(graphlab::iarchive& iarc) {
        iarc >> counter;
    }

    bool empty() const {
        return counter.empty();
    }

    Counter& operator+=(const Counter& other) {
        for (map_t::const_iterator it = other.counter.begin();
                it != other.counter.end(); it++)
            counter[it->first] += it->second;
        return *this;
    }
};

typedef Counter VertexData;
typedef graphlab::empty EdgeData; // no edge data
typedef Counter MessageData;

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<VertexData, EdgeData> graph_type;

inline uint16_t select_prob(uint16_t count, double prob = 1-RESET_PROB) {
    double remain = count * prob;
    uint16_t new_count = (uint16_t) remain;
    new_count += (graphlab::random::rand01() < remain-new_count) ? 1 : 0;
    return new_count;
}

class PreprocessProgram : public graphlab::ivertex_program<graph_type,
    graphlab::empty, MessageData> {
private:
    Counter walkers;

public:
    void init(icontext_type& context, const vertex_type& vertex,
            const message_type& msg) {
        if (phase == 0) {
            if (context.iteration() == 0) {
                walkers = Counter();
                if (sources == NULL || sources->find(vertex.id()) != sources->end())
                    walkers.counter[vertex.id()] = num_walkers;
            } else
                walkers = msg;
        } else { // if (phase == 1)
            walkers = msg;
            context.stop();
        }
    }

    edge_dir_type gather_edges(icontext_type& context,
            const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

    graphlab::empty gather(icontext_type& context, const vertex_type& vertex,
            edge_type& edge) const {
        return graphlab::empty();
    }

    void apply(icontext_type& context, vertex_type& vertex,
            const gather_type& total) {
        if (phase == 1) {
            vertex.data() = walkers;
            return;
        }
        vertex.data() += walkers;
        for (map_t::iterator it = walkers.counter.begin(); it != walkers.counter.end(); ) {
            uint16_t new_count = select_prob(it->second);
            if (new_count > 0) {
                it->second = new_count;
                it++;
            } else
                it = walkers.counter.erase(it);
        }
    }

    edge_dir_type scatter_edges(icontext_type& context,
            const vertex_type& vertex) const {
        if (phase == 1)
            return graphlab::NO_EDGES;

        if (vertex.num_out_edges() > 0 && !walkers.empty())
            return graphlab::OUT_EDGES;
        else if (!walkers.empty() && context.iteration() < (int) niters-1) {
            for (map_t::const_iterator it = walkers.counter.begin(); it != walkers.counter.end(); it++) {
                Counter msg;
                msg.counter[it->first] = it->second;
                context.signal_vid(it->first, msg);
            }
            return graphlab::NO_EDGES;
        } else
            return graphlab::NO_EDGES;
    }

    void scatter(icontext_type& context, const vertex_type& vertex,
            edge_type& edge) const {
        if (context.iteration() < (int) niters-1) {
            Counter msg;
            for (map_t::const_iterator it = walkers.counter.begin(); it != walkers.counter.end(); it++) {
                uint16_t count = select_prob(it->second, 1.0 / vertex.num_out_edges());
                if (count > 0)
                    msg.counter[it->first] = count;
            }
            if (!msg.empty())
                context.signal(edge.target(), msg);
        }
    }

    void save(graphlab::oarchive& oarc) const {
        oarc << walkers;
    }

    void load(graphlab::iarchive& iarc) {
        iarc >> walkers;
    }
};

typedef graphlab::synchronous_engine<PreprocessProgram> engine_type;

void collect_results(engine_type::icontext_type& context,
        graph_type::vertex_type& vertex) {
    for (map_t::const_iterator it = vertex.data().counter.begin(); it !=
            vertex.data().counter.end(); it++) {
        Counter msg;
        msg.counter[vertex.id()] = it->second;
        context.signal_vid(it->first, msg);
    }
    vertex.data() = VertexData();
}

bool compare(const std::pair<graphlab::vertex_id_type, uint16_t>& firstElem,
        const std::pair<graphlab::vertex_id_type, uint16_t>& secondElem) {
      return firstElem.second > secondElem.second;
}

struct pagerank_writer {
    size_t topk;

    pagerank_writer(size_t topk) : topk(topk) { }

    std::string save_vertex(graph_type::vertex_type vertex) {
        std::stringstream strm;
        if (!vertex.data().empty()) {
            strm << vertex.id();
            std::vector<std::pair<graphlab::vertex_id_type, uint16_t> >
                result(vertex.data().counter.begin(), vertex.data().counter.end());
            std::sort(result.begin(), result.end(), compare);
            size_t len = std::min(topk, result.size());
            strm << " " << len;
            for (size_t i = 0; i < len; i++)
                strm << " " << result[i].first;
            strm << std::endl;
        }
        return strm.str();
    }
    std::string save_edge(graph_type::edge_type e) { return ""; }
};

int main(int argc, char** argv) {
    // Initialize control plane using mpi
    graphlab::mpi_tools::init(argc, argv);
    graphlab::distributed_control dc;
    global_logger().set_log_level(LOG_INFO);

    // Parse command line options -----------------------------------------------
    graphlab::command_line_options clopts("Multi-Source "
            "Personalized PageRank algorithm.");
    std::string graph_dir;
    std::string format = "snap";
    clopts.attach_option("graph", graph_dir,
            "The graph file.  If none is provided "
            "then a toy graph will be created");
    clopts.add_positional("graph");
    clopts.attach_option("format", format, "The graph file format");
    size_t powerlaw = 0;
    clopts.attach_option("powerlaw", powerlaw,
            "Generate a synthetic powerlaw out-degree graph. ");
    clopts.attach_option("R", num_walkers,
            "Number of walkers for each vertex");
    niters = 10;
    clopts.attach_option("niters", niters,
            "Number of iterations");
    std::string bin_prefix;
    clopts.attach_option("bin_prefix", bin_prefix,
            "If set, will save the whole graph to a sequence "
            "of binary files with prefix bin_prefix");
    std::string ppr_prefix;
    clopts.attach_option("ppr_prefix", ppr_prefix,
            "If set, will save the resultant PPR to a sequence "
            "of human readable files with prefix ppr_prefix");
    size_t topk = 100;
    clopts.attach_option("topk", topk,
            "Output top-k elements of PPR vectors");
    std::string sources_file;
    clopts.attach_option("sources_file", sources_file,
            "The file contains all sources.");

    if(!clopts.parse(argc, argv)) {
        dc.cout() << "Error in parsing command line arguments." << std::endl;
        return EXIT_FAILURE;
    }

    clopts.get_engine_args().set_option("max_iterations", niters);

    // Build the graph ----------------------------------------------------------
    double start_time = graphlab::timer::approx_time_seconds();
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
    double runtime = graphlab::timer::approx_time_seconds() - start_time;
    dc.cout() << "Finished loading graph in " << runtime
        << " seconds." << std::endl;

    if (sources_file.length() > 0) {
        sources = new boost::unordered_set<graphlab::vertex_id_type>();
        std::ifstream fin(sources_file.c_str());
        int num_sources;
        fin >> num_sources;
        for (int i = 0; i < num_sources; i++) {
            graphlab::vertex_id_type vid;
            fin >> vid;
            sources->insert(vid);
        }
    }

    // Running The Engine -------------------------------------------------------
    engine_type engine(dc, graph, clopts);
    engine.signal_all();
    phase = 0;
    engine.start();
    runtime = engine.elapsed_seconds();
    dc.cout() << "Finished running engine in " << runtime
        << " seconds." << std::endl;

    engine.transform_vertices(collect_results);
    phase = 1;
    engine.start();
    runtime = engine.elapsed_seconds();
    dc.cout() << "Finished running engine in " << runtime
        << " seconds." << std::endl;

    if (sources)
        delete sources;

    // Save the final graph -----------------------------------------------------
    if (bin_prefix != "") {
        graph.save_binary(bin_prefix);
    }
    if (ppr_prefix != "") {
        graph.save(ppr_prefix, pagerank_writer(topk),
                false,    // do not gzip
                true,     // save vertices
                false);   // do not save edges
    }

    // Tear-down communication layer and quit -----------------------------------
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}

