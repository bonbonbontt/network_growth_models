import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns


def ig_to_nx(ig_graph, directed=False, nodes=None):
    """map igraph Graph object to networkx Graph object"""
    g = nx.DiGraph() if directed else nx.Graph()
    nodes = nodes if nodes else ig_graph.vs
    edges = ig_graph.induced_subgraph(nodes).es if nodes else ig_graph.es
    for node in nodes: g.add_node(node.index, **node.attributes())
    for edge in edges: g.add_edge(edge.source, edge.target)
    return g

def get_discrete_distribution(data):
    """given array of numberes, compute its discrete distribution"""
    counter = Counter(data)
    vals, freq = map(np.array, zip(*sorted(counter.items())))
    prob = freq/float(np.sum(freq))
    return vals, prob

def get_fitted_parameters():
    """return fitted parameters for hz, arw, sk, hk, dms, ff, lapa, hpa]"""
    with open('data/params.pkl', 'r') as f:
        params = pickle.load(f)
    return params

def get_fits_info():
    return pd.read_pickle('data/wsdm_fits_data.pkl')

if __name__ == '__main__':
    params = get_fitted_parameters()
    info = get_fits_info()
