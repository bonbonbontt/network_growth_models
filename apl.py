import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns


def get_graphs():
	networks = {}
	for gpath in paths:
		fname = gpath.rsplit('/', 1)[-1].split('.')[0]
		networks[fname] = ig.Graph.Read_Pickle(gpath)
		print fname
		print networks[fname].summary()
	return networks

def apl(graph, N, deg_mode = 'ALL'):
	nids = np.random.choice(graph.vs.indices, size=N)
	allnids = graph.vs.indices
	lengths = np.array(graph.shortest_paths(source=nids, mode=deg_mode))

	# write distances to file
	full_name = output_path + '/' + fname + '.txt'
	f = open(full_name, 'w')
	for row in range(len(nids)):
		for col in range(len(allnids)):
			f.write(str(nids[row]) + " " + str(allnids[col]) + " " + str(lengths[row][col]) + "\n")
	f.close()


if __name__ == '__main__':
	path = 'data/networks'
	output_path = 'data/path_length_outputs'
	paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
	networks = get_graphs()
	for fname in networks:
		N = int(len(networks[fname].vs) * 0.01)
		apl(networks[fname], N)




