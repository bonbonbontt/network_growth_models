import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns

def _get_graph_params(graph, time_attr, discard_nids, init_nids, post_nids, use_mean_outdeg=False, debug=True, attrs=None):
    get_summary = lambda g: g.summary().rsplit('--', 1)[0]

    discard_nids = set(discard_nids)
    all_nodes = graph.vs.select(list(post_nids))
    chunk_outdeg_map = defaultdict(list)
    chunk_attr_list = defaultdict(list)
    if debug: print ("next chunk + outdeg")

    # compute "actual" chunks (use main graph) and mean outdegree
    for node in all_nodes:
        nbors = [nbor for nbor in node.neighbors(mode='OUT') if nbor.index not in discard_nids]
        ta = node[time_attr]
        if ta != ta: continue
        chunk_outdeg_map[node[time_attr]].append(len(nbors))
        if attrs: chunk_attr_list[node[time_attr]].append(node[attrs])

    chunk_attr_seq = []
    for k in sorted(chunk_attr_list):
        chunk_attr_seq.append(chunk_attr_list[k])

    chunk_deg_seq = [(k, len(v), np.mean(v)) for k,v in sorted(chunk_outdeg_map.items())]
    time_keys, chunks, outdegs = zip(*chunk_deg_seq)
    if use_mean_outdeg: outdegs = np.array([np.mean(graph.outdegree())]*len(outdegs))

    # subgraphs
    if debug: print ("next subgraphs")
    gpre = graph.subgraph(list(init_nids))
    gpost = graph.subgraph(list(post_nids))
    geval = graph.subgraph(list(init_nids.union(post_nids)))

    return {
        'graph': geval,
        'gpre': gpre,
        'gpost': gpost,
        'time_keys': time_keys,
        'chunk_sizes': chunks,
        'mean_outdegs': outdegs,
        'mean_chunk_size': np.mean(chunks),
        'mean_outdeg': np.mean(outdegs),
        'N': len(geval.vs),
        'nids_order': None,
        'chunk_sampler': chunk_attr_seq
    }

def get_discard_graph_params(graph, time_attr, discard_pct, init_pct, debug=True, use_mean_outdeg=False, attrs=None, min_bfs_size=10, eval_pct=1.):
    # sort nids
    nids, _ = zip(*sorted(zip(graph.vs.indices, graph.vs[time_attr]), key=lambda t: t[-1]))

    discard_idx = int(round(discard_pct*len(nids)))
    use_nids = nids[discard_idx:]
    use_nids_iter = iter(use_nids)

    # construct initial subgraph using BFS
    N_gpre = int(round(init_pct*len(nids)))
    if debug: print ("initial graph size: {}".format(N_gpre))
    init_nids =  set()

    while len(init_nids) <= N_gpre:
        nid = random.choice(use_nids)
        if nid in init_nids: continue

        visited_nids = []

        for node in graph.bfsiter(nid, mode='OUT'):
            visited_nids.append(node.index)
            if len(visited_nids) > N_gpre: break

        if len(visited_nids) < min(min_bfs_size, N_gpre):
            continue # ignore if BFS too small

        num_add = min(N_gpre-len(init_nids)+1, len(visited_nids))
        init_nids.update(visited_nids[:num_add])

    if debug: print ("bfs, next graph_params")
    discard_nids = set(nids[:discard_idx])-init_nids

    discard_and_init = discard_nids.union(init_nids)
    post_nids = [n for n in nids if n not in discard_and_init]
    post_idx = int(round(eval_pct*len(post_nids)))
    post_nids = set(post_nids[:post_idx+1])

    return _get_graph_params(graph, time_attr, discard_nids, init_nids, post_nids, debug=debug, use_mean_outdeg=use_mean_outdeg, attrs=attrs)