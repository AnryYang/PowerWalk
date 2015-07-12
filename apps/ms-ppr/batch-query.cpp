#include <vector>
#include <string>
#include <fstream>
#include <map>

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include <graphlab.hpp>

typedef float float_t;
// Global random reset probability
const float_t RESET_PROB = 0.15;
const float_t EPSILON = 1e-6;
int niters;
boost::unordered_set<graphlab::vertex_id_type> *sources = NULL;

enum phase_t {INIT_GRAPH, COMPUTE};
phase_t phase = INIT_GRAPH;

typedef boost::unordered_map<graphlab::vertex_id_type, uint16_t> map_t;
typedef boost::unordered_map<graphlab::vertex_id_type, float_t> ppr_t;

struct VertexData {
    ppr_t ppr;
    graphlab::dense_bitset schedule;

    VertexData() : ppr(), schedule(niters) {}

    void save(graphlab::oarchive& oarc) const {
        oarc << schedule;
        oarc << ppr;
    }

    void load(graphlab::iarchive& iarc) {
        if (phase == INIT_GRAPH) {
            map_t counter;
            iarc >> counter;
            float_t sum = 0.0;
            for (map_t::const_iterator it = counter.begin(); it != counter.end(); it++)
                sum += it->second;
            for (map_t::const_iterator it = counter.begin(); it != counter.end(); it++)
                ppr[it->first] = it->second / sum;
        } else {
            iarc >> schedule;
            iarc >> ppr;
        }
    }
};

typedef graphlab::empty EdgeData; // no edge data

struct max_combiner : public graphlab::IS_POD_TYPE {
    float_t value;
    max_combiner() : value(0.0) {}
    max_combiner(float_t v) : value(v) {}
    max_combiner& operator+=(const max_combiner& other) {
        if (other.value > value)
            value = other.value;
        return *this;
    }
};

struct ppr_gather_t {
    ppr_t ppr;

    ppr_gather_t() : ppr() { }

    ppr_gather_t(ppr_t ppr) : ppr(ppr) { }

    void save(graphlab::oarchive& oarc) const {
        oarc << ppr;
    }

    void load(graphlab::iarchive& iarc) {
        iarc >> ppr;
    }

    bool empty() const {
        return ppr.empty();
    }

    ppr_gather_t& operator+=(const ppr_gather_t& other) {
        for (ppr_t::const_iterator it = other.ppr.begin();
                it != other.ppr.end(); it++)
            ppr[it->first] += it->second;
        return *this;
    }
};

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<VertexData, EdgeData> graph_type;

class ForwardExpansion : public graphlab::ivertex_program<graph_type,
    graphlab::empty, max_combiner> {
private:
    float_t flow;

public:
    void init(icontext_type& context, const vertex_type& vertex,
            const message_type& msg) {
        if (context.iteration() == 0) {
            if (sources == NULL || sources->find(vertex.id()) != sources->end())
                flow = 1.0;
        } else
            flow = msg.value;
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
        if (flow > EPSILON) {
            for (int i = context.iteration(); i < niters; i++)
                vertex.data().schedule.set_bit_unsync(i);
            flow = flow * (1-RESET_PROB);
            if (vertex.num_out_edges() > 0)
                flow /= vertex.num_out_edges();
        }
    }

    edge_dir_type scatter_edges(icontext_type& context,
            const vertex_type& vertex) const {
        if (flow > EPSILON)
            return graphlab::OUT_EDGES;
        else
            return graphlab::NO_EDGES;
    }

    void scatter(icontext_type& context, const vertex_type& vertex,
            edge_type& edge) const {
        context.signal(edge.target(), max_combiner(flow));
    }

    void save(graphlab::oarchive& oarc) const {
        oarc << flow;
    }

    void load(graphlab::iarchive& iarc) {
        iarc >> flow;
    }
};

class BackwardExpansion : public graphlab::ivertex_program<graph_type,
    ppr_gather_t>, graphlab::IS_POD_TYPE {
public:
    void init(icontext_type& context, const vertex_type& vertex,
            const message_type& msg) {
    }

    edge_dir_type gather_edges(icontext_type& context,
            const vertex_type& vertex) const {
        if (vertex.data().schedule.get(niters-context.iteration()-1))
            return graphlab::OUT_EDGES;
        else
            return graphlab::NO_EDGES;

    }

    gather_type gather(icontext_type& context, const vertex_type& vertex,
            edge_type& edge) const {
        return ppr_gather_t(edge.target().data().ppr);
    }

    void apply(icontext_type& context, vertex_type& vertex,
            const gather_type& total) {
        if (!total.empty()) {
            vertex.data().ppr = total.ppr;
            float_t c = (1-RESET_PROB) / vertex.num_out_edges();
            for (ppr_t::iterator it = vertex.data().ppr.begin();
                    it != vertex.data().ppr.end(); it++)
                it->second *= c;
            vertex.data().ppr[vertex.id()] += RESET_PROB;
        }
        if (context.iteration() < niters-1 &&
                vertex.data().schedule.get(niters-context.iteration()-2))
            context.signal(vertex);
    }

    edge_dir_type scatter_edges(icontext_type& context,
            const vertex_type& vertex) const {
        return graphlab::NO_EDGES;
    }

    void scatter(icontext_type& context, const vertex_type& vertex,
            edge_type& edge) const { }
};

bool compare(const std::pair<graphlab::vertex_id_type, float_t>& firstElem,
        const std::pair<graphlab::vertex_id_type, float_t>& secondElem) {
      return firstElem.second > secondElem.second;
}

struct pagerank_writer {
    size_t topk;

    pagerank_writer(size_t topk) : topk(topk) { }

    std::string save_vertex(graph_type::vertex_type vertex) {
        std::stringstream strm;
        if (!vertex.data().ppr.empty()) {
            strm << vertex.id();
            std::vector<std::pair<graphlab::vertex_id_type, float_t> >
                result(vertex.data().ppr.begin(), vertex.data().ppr.end());
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
    clopts.attach_option("graph", graph_dir,
            "The binary graph file that contains preprocessed PPR."
            "Must be provided.");
    clopts.add_positional("graph");
    niters = 10;
    clopts.attach_option("niters", niters,
            "Number of iterations");
    std::string saveprefix;
    clopts.attach_option("saveprefix", saveprefix,
            "If set, will save the whole graph to a "
            "sequence of files with prefix saveprefix");
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
    phase = INIT_GRAPH;
    graph_type graph(dc, clopts);
    graph.load_binary(graph_dir);
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
    phase = COMPUTE;
    graphlab::synchronous_engine<ForwardExpansion> engine(dc, graph, clopts);
    engine.signal_all();
    engine.start();
    runtime = engine.elapsed_seconds();
    dc.cout() << "Finished forward expansion in " << runtime << " seconds." << std::endl;

    graphlab::synchronous_engine<BackwardExpansion> engine2(dc, graph, clopts);
    engine2.signal_all();
    engine2.start();
    runtime = engine2.elapsed_seconds();
    dc.cout() << "Finished backward expansion in " << runtime << " seconds." << std::endl;

    if (sources)
        delete sources;

    // Save the final graph -----------------------------------------------------
    if (saveprefix != "") {
        graph.save(saveprefix, pagerank_writer(topk),
                false,    // do not gzip
                true,     // save vertices
                false);   // do not save edges
    }

    // Tear-down communication layer and quit -----------------------------------
    graphlab::mpi_tools::finalize();
    return EXIT_SUCCESS;
}

