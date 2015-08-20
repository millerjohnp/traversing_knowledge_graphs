import graph_traversal as gt

g = gt.Graph()
g.add_edge("1", "edge1", "2")
g.add_edge("2", "edge2", "3")
g.add_edge("3", "edge3", "4")
g.add_edge("1", "edge1", "12")
g.add_edge("12", "edge2", "13")
g.add_edge("13", "edge3", "1")
g.add_edge("1", "edge1", "9")
g.add_edge("2", "edge2", "10")
g.add_edge("3", "edge3", "11")

# ensure model removes duplicates
g.add_edge("1", "edge1", "20")
g.add_edge("20", "edge2", "30")
g.add_edge("30", "edge3", "4")

# add a bunch of spurious edges
g.add_edge("1", "edge2", "4")
g.add_edge("1", "edge2", "5")
g.add_edge("5", "edge5", "6")
g.add_edge("5", "edge3", "2")
g.add_edge("6", "edge1", "3")
g.add_edge("2", "edge2", "7")
g.add_edge("6", "edge5", "8")
g.add_edge("11", "edge3", "6")
g.add_edge("8", "edge4", "11")

print "Testing Path Traversal from node 1: (Should be 1, 4, 11)"
path = ['edge1', 'edge2', 'edge3']

targets = g.path_traversal("1", path)

for target in targets:
	print target

print "Testing MC Estimate for 1 Target, 1 Path (Should be 2, 1)"
g = gt.Graph()
g.add_edge("1", "edge1", "2")
print g.random_walk_probs("1", ["edge1"])

print "Testing MC Estimate for 1 Target, 1 Path (Should be 4, 1)"
g = gt.Graph()
g.add_edge("1", "edge1", "2")
g.add_edge("2", "edge2", "3")
g.add_edge("3", "edge3", "4")

print g.random_walk_probs("1", path)

print "Testing MC Estimate for 1 Target, 2 Paths (Should be 4, 1)"
g = gt.Graph()
g.add_edge("1", "edge1", "2")
g.add_edge("2", "edge2", "3")
g.add_edge("3", "edge3", "4")
g.add_edge("1", "edge1", "2a")
g.add_edge("2a", "edge2", "3a")
g.add_edge("3a", "edge3", "4")

print g.random_walk_probs("1", path)

print "Testing MC Estimate for 2 Targets, 1 Paths (Should be t1, t2, 0.5)"
g = gt.Graph()
g.add_edge("1", "edge1", "2")
g.add_edge("2", "edge2", "3")
g.add_edge("3", "edge3", "t1")
g.add_edge("1", "edge1", "2a")
g.add_edge("2a", "edge2", "3a")
g.add_edge("3a", "edge3", "t2")

print g.random_walk_probs("1", path)


print "Testing MC Estimate for 2 Targets, 2 Paths. One Dead End(Should be t1, t2= 0.5)"
g = gt.Graph()
g.add_edge("1", "edge1", "2")
g.add_edge("2", "edge2", "3")
g.add_edge("3", "edge3", "t1")
g.add_edge("2", "edge2", "3a")
g.add_edge("3a", "edge3", "t2")

g.add_edge("1", "edge1", "dead_end")
print g.random_walk_probs("1", path)

print "Testing MC Estimate for 1 Targets, 3 Deep BST. (Should be t1 = 1.)"
g = gt.Graph()
g.add_edge("1", "e", "2")
g.add_edge("1", "e", "3")
g.add_edge("2", "e", "4")
g.add_edge("2", "e", "5")
g.add_edge("3", "e", "6")
g.add_edge("3", "e", "7")
g.add_edge("4", "e", "t1")
g.add_edge("5", "e", "t1")
g.add_edge("6", "e", "t1")
g.add_edge("7", "e", "t1")

print g.random_walk_probs("1", ['e', 'e','e'])

print "Testing MC Estimate for 2 Targets, 3 Deep BST. (Should be t1 = 0.5, t2 = 0.5)"
g = gt.Graph()
g.add_edge("1", "e", "2")
g.add_edge("1", "e", "3")
g.add_edge("3", "e", "7")
g.add_edge("7", "e", "t2")
g.add_edge("2", "e", "4")
g.add_edge("2", "e", "5")
g.add_edge("2", "e", "6")
g.add_edge("4", "e", "t1")
g.add_edge("5", "e", "t1")
g.add_edge("6", "e", "t1")

print g.random_walk_probs("1", ['e', 'e','e'])

