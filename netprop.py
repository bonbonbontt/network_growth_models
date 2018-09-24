import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns
import utils

def get_node_path_length(graph, nid, deg_mode='ALL'):
    """compute distances to all nodes from nid"""
    lengths = graph.as_undirected().shortest_paths(source=nid, mode=deg_mode)
    counter = Counter(lengths)
    counter.pop(np.inf, None); counter.pop(0, None);
    return dict(counter)
    
def get_degree_distribution(graph, deg_mode='IN'):
    return utils.get_discrete_distribution(graph.degree(mode=deg_mode))

if __name__ == '__main__':
    pass