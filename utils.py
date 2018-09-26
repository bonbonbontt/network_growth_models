import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns
import baselines
import misc

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

#########
######### code to generate fitted networks
#########

def get_input(graph, time_attr, discard_pct, init_pct, cat_attr=None):
    return misc.get_discard_graph_params(graph, time_attr, discard_pct,
                                         init_pct, debug=False, attrs=cat_attr)

def generate_graph(model_name, params, input_data, debug=False):
    gpre, chunks, outdegs = [input_data[k] for k in ['gpre', 'chunk_sizes', 'mean_outdegs']]
    if model_name in ['sk', 'hz', 'hk']:
        cls = baselines.baselines_dict[model_name]
        g = cls(params, gpre=gpre, debug=debug)
    elif model_name == 'dms':
        g = cls['dms'](gpre=gpre, outpref=False, zero_appeal=params)
    else:
        # TODO: add ARW
        g = None
    g.add_nodes(chunks, outdegs)
    return g

def get_network_fit(model_name, dataset_name):
    """
    use fitted parameters and input to generate a network
    using model_name for dataset_name.
    ------
    model: dms, rl, hk, sk, hz, arw
    dataset: (small) hepph, judicial, acl (large) aps, patetnts, semantic
    """
    dset = ig.Graph.Read_Pickle('data/networks/{}.pkl'.format(dataset_name))

    # get parameters
    params_data = get_fitted_parameters()
    fitted_params = params_data[model_name][dataset_name]

    # get input data
    time_attr = 'year2' if dataset_name in ['patents'] else 'time'
    cat_attr = None
    discard_pct = 0.1 if dataset_name in ['patents'] else 0.05
    initial_pct = 0.01
    input_data = get_input(dset, time_attr, discard_pct, initial_pct, cat_attr=cat_attr)

    graph = generate_graph(model_name, fitted_params, input_data)

    return {
        'dset': dset,
        'input': input_data,
        'graph': graph
    }

if __name__ == '__main__':
    d = get_network_fit('hk', 'hepph')