import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import igraph as ig
import netprop
import utils

"""
This script takes two arguments
- graph path (path to pickled graph)
- out path (path to write path lengths)
"""

def run(g, fname, update_every=None):
	# load input
	f_out = open(fname, 'a')
	g = g.as_undirected()
	N = len(g.vs)
	nids = g.vs.indices[:]
	random.shuffle(nids)
	sampled_nids = set()

	while (len(sampled_nids) < N:
		# sample a node, compute its lengths
		nid = nids.pop()
		pl = netprop.get_node_path_length(g, nid)

		# update file
		for path_lenth, freq in pl.items():
			f_out.write('{} {} {}\n'.format(nid, path_lenth, freq))

		sampled_nids.add(nid)

		if update_every and (len(sampled_nids) % update_every == 0):
			print "{}/{} nodes done".format(len(sampled_nids), N)

	close(f_out)
	return input

if __name__ == '__main__':
	assert len(sys.argv) == 3, "enter arguments"
	_, graph_path, out_path = sys.argv

	g = ig.Graph.Read_Pickle(graph_path)
	run(g, out_path, 100)
