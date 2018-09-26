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

def update_ax(ax, title=None, xlabel=None, ylabel=None, legend_loc='best', despine=True):
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend_loc: ax.legend(loc=legend_loc)
    if despine: sns.despine(ax=ax)
    return ax

def get_spaced_data_generator(fname, proc=None):
    with open(fname, 'r') as f:
        for line in f:
            splitted_line = line.split(' ')
            if proc: splitted_line = proc(splitted_line)
            yield splitted_line

if __name__ == '__main__':
    pass