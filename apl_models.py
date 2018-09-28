import pickle, json, os, sys, csv, random, operator
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import seaborn as sns
import baselines
import misc
import utils
import apl
import os

'''
Generate graphs
'''
def apl_model(model_names, dataset_names, N):
	for i in range(N):
		for model in model_names:
			for dataset in dataset_names:
				_apl_model(model, dataset, i)

def _apl_model(model_name, dname, n):
	g = utils.get_network_fit(model_name, dname)['graph'].g
	fname = 'data/models_apl_outputs/graphs/%s_%s_%d.pkl'%(model_name, dname, (n+1))
	fname = "test"
	ig.Graph.write_pickle(g, fname=fname)
	print (fname + ' done!')

'''
Run apl each graph
'''
def run_model():
	# get input file list
	in_path = "data/models_apl_outputs/graphs"
	all_in_files = os.listdir(in_path)

	# get output file list
	out_path = "data/models_apl_outputs/apl_outputs"
	all_out_files = os.listdir(out_path)

	if len(all_in_files) == len(all_out_files):
		print("All done!")

	for f in all_in_files:
		fname = f.split('.')[0] + '.txt'
		if fname not in all_out_files:
			print('Processing ' + fname)
			graph_path = in_path + '/' + f
			out_path = "data/models_apl_outputs/apl_outputs/%s"%fname
			g = ig.Graph.Read_Pickle(graph_path)
			apl.run(g, out_path, 100)
			print(fname + " done!")
			break

'''
Put three rounds of outputs into one file
'''
def serialize(model_names, dataset_names):
	for mname in model_names:
		for dname in dataset_names:
			prefix = mname + '_' + dname
			print ('serializing %s'%prefix)
			all_files = os.listdir('data/models_apl_outputs/apl_outputs')
			f = open('data/models_apl_outputs/%s_all.txt'%prefix, 'w')
			for fname in all_files:
				if prefix in fname:
					fin = open('data/models_apl_outputs/apl_outputs/' + fname)
					f.write(fin.read())
					fin.close()
					print (fname + " done!")


if __name__ == '__main__':
	model_names = ['hk', 'hz', 'sk']
	dataset_names = ['judicial', 'hepph', 'acl']
	# apl_model(model_names,dataset_names,3)
	# run_model()
	serialize(model_names, dataset_names)
