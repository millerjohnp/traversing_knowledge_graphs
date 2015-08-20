
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <boost/python.hpp>
using namespace std;

class Graph
{
	public:
		// string name;
		map<string, map<string, vector<string> > > edges;
		map<string, set<string> > relation_targets;
	public:
		Graph();
		void addEdge(string start, string edge_type, string end);
		boost::python::list pathTraversal(string start, boost::python::list& path);
		boost::python::list approx_pathTraversal(string start, boost::python::list& path, 
																				 		 unsigned int maximum_degree);
		boost::python::dict exactRandomWalkProb(string start, boost::python::list& path);
		bool is_Trivial_Query(string start, boost::python::list &path);
};

Graph::Graph()
{
	edges = map<string, map<string, vector<string> > > ();
	srand (time(NULL));
}

void Graph::addEdge(string start, string edge_type, string end)
{
	edges[start][edge_type].push_back(end);
	relation_targets[edge_type].insert(end);
}

/**
 * Return a boost::python::list containing all of the unique enitites reachable
 * along edge-type sequence in PATH beginning at entity START
 */
boost::python::list Graph::pathTraversal(string start, boost::python::list& path)
{
	return Graph::approx_pathTraversal(start, path, UINT_MAX);
}

boost::python::list Graph::approx_pathTraversal(string start, boost::python::list& path, 
																				 				unsigned int maximum_degree)
{
	// maintain two queues... one for the current time step and one
	// for the next time step
	vector<string> visited = vector<string>();
	set<string> next = set<string>();

	// add the start node to begin the process
	visited.push_back(start);

	int path_length = boost::python::len(path);
	for(int i = 0; i < path_length; i++) {
		next.clear();
		string curr_edge_type = boost::python::extract<string>(path[i]);
		while(!visited.empty()) {
			string curr_node = visited.back();
			visited.pop_back();

			map<string, vector<string> > curr_edge_set = edges[curr_node];
			if(curr_edge_set.find(curr_edge_type) != curr_edge_set.end()){ 
				vector<string> next_nodes = curr_edge_set[curr_edge_type];

				// if we have to subsample, at least randomize the first MAXIMUM_DEGREE 
				// elements
				if(next_nodes.size() > maximum_degree) {
					int n = next_nodes.size();
					for(unsigned int idx=0; idx < maximum_degree; idx++){
						swap(next_nodes[idx], next_nodes[rand() % (n - idx) + idx]);
					}
					next.insert(next_nodes.begin(), next_nodes.begin() + maximum_degree);
				} else {
					next.insert(next_nodes.begin(), next_nodes.end());
				}
			}
		}
		visited.assign(next.begin(), next.end());
	}

  boost::python::list returnList;
  for(set<string>::iterator it = next.begin(); it != next.end(); ++it){
    returnList.append(*it);
  }
  return returnList;
}

bool Graph::is_Trivial_Query(string start, boost::python::list &path){
	// at each step, compare the beam with everything else and then quit
	// and return false if the size of the beam type matches with everything!
	vector<string> visited = vector<string>();
	set<string> next = set<string>();

	// add the start node to begin the process
	visited.push_back(start);

	int path_length = boost::python::len(path);

	for(int step = 0; step < path_length; step++){
		next.clear();
		string curr_edge_type = boost::python::extract<string>(path[step]);
		while(!visited.empty()) {
			string curr_node = visited.back();
			visited.pop_back();
			map<string, vector<string> > curr_edge_set = edges[curr_node];
			if(curr_edge_set.find(curr_edge_type) != curr_edge_set.end()){ 
				vector<string> next_nodes = curr_edge_set[curr_edge_type];
				next.insert(next_nodes.begin(), next_nodes.end());
			}
		}

		if(next == relation_targets[curr_edge_type]){
			return true;
		}

		visited.assign(next.begin(), next.end());
	}
	return false; // this query is not trivial
}

/**
 * Return a boost::python::dict with the random walk probabilities of 
 * each possible target
 */
boost::python::dict Graph::exactRandomWalkProb(string start, boost::python::list &path)
{
	// stores the currently enqueued nodes + number of paths to reach
	// each one
	map<string, double> curr_nodes = map<string, double>();

	curr_nodes[start] = 1.0; 

	int path_length = boost::python::len(path);
	for(int i = 0; i < path_length; i++) {
		string curr_edge_type = boost::python::extract<string>(path[i]);
		map<string, double> next_nodes = map<string, double>();
		for(map<string, double>::iterator curr_node = curr_nodes.begin(); 
			  curr_node != curr_nodes.end(); curr_node++) {
			map<string, vector<string> > curr_edge_set = edges[curr_node->first];

			if(curr_edge_set.find(curr_edge_type) != curr_edge_set.end()){ 
				vector<string> nodes = curr_edge_set[curr_edge_type];
				double prev_prob = curr_node->second / float(nodes.size());

				for(vector<string>::iterator node_it = nodes.begin(); 
						node_it!= nodes.end(); node_it++){
					if(next_nodes[*node_it] == 0) {
						next_nodes[*node_it] = prev_prob;
					}
					else{
						next_nodes[*node_it] += prev_prob;
					}
				}
			}
		}
		curr_nodes = next_nodes;
	}

	// convert map to python dictionary
	boost::python::dict dictionary;
	for(map<string, double>::iterator curr_node = curr_nodes.begin(); 
			  curr_node != curr_nodes.end(); curr_node++) {
		dictionary[curr_node->first] = curr_node->second; // no renormalization
	}
	return dictionary;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(graph_traversal)
{
  class_<Graph>("Graph")
    .def("add_edge", &Graph::addEdge)
    .def("path_traversal", &Graph::pathTraversal)
    .def("approx_path_traversal", &Graph::approx_pathTraversal)
    .def("exact_random_walk_probs", &Graph::exactRandomWalkProb)
    .def("is_trivial_query", &Graph::is_Trivial_Query);
  ;
};

