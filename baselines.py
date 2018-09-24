import igraph as ig
import numpy as np
import random
import time
from collections import defaultdict, Counter, deque

random.seed(time.time())

class Barabasi(object):
	"""
	Linear time BA w. chunking + initial graph
	Dorogotsev Samukhim Mendes model if zero appeal nonzero and graph directed
	"""

	def __init__(self, gpre=None, outpref=True, zero_appeal=0, directed=True, debug=False, er_n0=100, er_m=5):
		if gpre: assert gpre.is_directed() == directed
		self.g = gpre.copy() if gpre else None
		self.directed = directed
		self.er_n0 = er_n0
		self.er_m = er_m
		self.outpref = outpref # probability ~ indegree or degree
		self.zero_appeal = zero_appeal
		self.setup()

	def setup(self):
		# setup initial graph
		if not self.g:
			p = min(1., self.er_m/float(self.er_n0-1))
			self.g = ig.Graph.Erdos_Renyi(self.er_n0, p=p, directed=self.directed)

		# add initial graph to M
		self.M = []
		for node in self.g.vs:
			nid = node.index

			if self.zero_appeal: # initial attractiveness
				self.M.extend([nid]*2*self.zero_appeal)

			for nbor in node.neighbors(mode='OUT'): # (out, in), (from, to)
				self.M.append(nid)
				self.M.append(nbor.index)

		# setup tracking
		self.start_idx = len(self.M)
		self.next_nid = len(self.g.vs)

	def add_chunk(self, chunk_size, m):
		for _c in xrange(chunk_size):

			max_nid = len(self.M)
			mi = int(m)
			mi = mi+1 if random.random() < m-mi else mi

			for _m in xrange(mi):
				self.M.append(self.next_nid)
				idx = random.randint(0, max_nid-1)
				if not self.outpref and idx % 2 == 0: idx += 1
				self.M.append(self.M[idx])

			if self.zero_appeal:
				self.M.extend([self.next_nid]*2*self.zero_appeal)

			self.next_nid += 1

	def add_nodes(self, chunk_seq, m_seq):
		for chunk, m in zip(chunk_seq, m_seq):
			self.add_chunk(chunk, m)
		self.update(sum(chunk_seq))

	def update(self, N):
		edges = list(zip(self.M[self.start_idx::2], self.M[self.start_idx+1::2]))
		self.g.add_vertices(N)
		self.g.add_edges(edges)
		self.g.simplify()
		self.start_idx = len(self.M)

class RelayLinking(Barabasi):

	def __init__(self, lbda, theta, iterated_relay=True, pa=True, gpre=None):
		self.lbda = lbda
		self.theta = theta
		self.iterated = iterated_relay
		self.pa = pa
		self.Rmax = 100
		super(RelayLinking, self).__init__(gpre=gpre)

	def setup(self):
		# setup initial graph
		if not self.g:
			p = min(1., self.er_m/float(self.er_n0-1))
			self.g = ig.Graph.Erdos_Renyi(self.er_n0, p=p, directed=self.directed)

		self.node_time_map = {}
		self.in_nbors = defaultdict(list)

		# add initial graph to M
		self.M = []
		for node in self.g.vs:
			nid = node.index
			self.node_time_map[nid] = 0 # initial
			if self.zero_appeal: # initial attractiveness
				self.M.extend([nid]*2*self.zero_appeal)

			for nbor in node.neighbors(mode='OUT'): # (out, in), (from, to)
				self.M.append(nid)
				self.M.append(nbor.index)
				self.in_nbors[nbor.index].append(nid)

		# setup tracking
		self.start_idx = len(self.M)
		self.next_nid = len(self.g.vs)
		self.node_time_map
		self.in_nbors

	def add_nodes(self, chunk_seq, m_seq):
		for cid, (chunk, m) in enumerate(zip(chunk_seq, m_seq), 1):
			self.add_chunk(chunk, m, cid)
		self.update(sum(chunk_seq))

	def add_node(self, m, max_nid, t):
		nbors = set()
		while len(nbors) < m:
			# get old paper
			u = random.randint(0, max_nid-1)
			if u % 2 == 0: u += 1
			u = self.M[u]
			t_u = self.node_time_map[u]

			# start relay
			relay, r = True, 0
			while relay and r < self.Rmax:
				r += 1
				dt = t-t_u

				exp_toss = random.random() < np.exp(-self.lbda*dt)
				if exp_toss or (not self.iterated and r > 1): relay = False
				bern_toss = random.random() < self.theta
				if not bern_toss: relay = False

				if not relay: continue

				i_ut = self.in_nbors[u]
				if i_ut: u = random.choice(i_ut)

			nbors.add(u)

		return list(nbors)

	def add_chunk(self, chunk_size, m, chunk_id):
		for _c in xrange(chunk_size):
			self.node_time_map [self.next_nid] = chunk_id

			max_nid = len(self.M)
			mi = int(m)
			mi = mi+1 if random.random() < m-mi else mi

			nbor_nids = self.add_node(mi, max_nid, chunk_id)

			for nbor_nid in nbor_nids:
				self.M.append(self.next_nid)
				self.M.append(nbor_nid)
				self.in_nbors[nbor_nid].append(self.next_nid)

			if self.zero_appeal:
				self.M.extend([self.next_nid]*2*zero_appeal)

			self.next_nid += 1

class Holmes(Barabasi):
	"""
	Holmes model
	p: P(TF step)
	Linear time BA w. chunking + initial graph
	"""
	def __init__(self, p, gpre=None, outpref=True, debug=False,
				 zero_appeal=False, directed=True, er_n0=50, er_m=5):
		self.p = p
		super(Holmes, self).__init__(gpre=gpre, outpref=outpref, directed=directed,
		 							 zero_appeal=zero_appeal, er_n0=er_n0, er_m=er_m)

	def setup(self):
		super(Holmes, self).setup()

		# track neighbors for TF step
		self.nbors_map = defaultdict(list)
		for node in self.g.vs:
			nid = node.index
			for nbor in node.neighbors(mode='ALL'):
				self.nbors_map[nid].append(nbor.index)

	def add_chunk(self, chunk_size, m):
		for _c in xrange(chunk_size):
			max_nid = len(self.M)
			_m = 0
			mi = int(m)
			mi = mi+1 if random.random() < m-mi else mi

			while _m < mi:
				_m += 1

				idx = random.randint(0, len(self.M)-1)
				if not self.outpref and idx % 2 == 0: idx += 1

				pa_nid = self.M[idx]

				self.M.append(self.next_nid)
				self.M.append(pa_nid)

				if random.random() < self.p and _m < mi and self.nbors_map[pa_nid]:
					_m += 1

					pa_nbor_nid = random.choice(self.nbors_map[pa_nid])

					self.M.append(self.next_nid)
					self.M.append(pa_nbor_nid)

					self.nbors_map[self.next_nid].append(pa_nbor_nid)
					self.nbors_map[pa_nbor_nid].append(self.next_nid)

				self.nbors_map[self.next_nid].append(pa_nid)
				self.nbors_map[pa_nid].append(self.next_nid)

			if self.zero_appeal: self.M.extend([self.next_nid, self.next_nid])
			self.next_nid += 1

	def add_nodes(self, chunk_seq, m_seq):
		for c,m in zip(chunk_seq, m_seq):
			self.add_chunk(c, m)
		self.update(sum(chunk_seq))

	def update(self, N):
		edges = list(zip(self.M[self.start_idx::2], self.M[self.start_idx+1::2]))
		self.g.add_vertices(N)
		self.g.add_edges(edges)
		self.g.simplify()
		self.start_idx = len(self.M)

class HereraZufiriaRandomWalk(object):

	def __init__(self, cc, gpre=None, directed=True, debug=True, debug2=False, er_m=5, er_n0=5, max_tries=50):
		self.cc = cc
		self.directed = directed
		self.gpre = gpre
		self.debug = debug
		self.m, self.n0 = er_m, er_n0
		self.g = ig.Graph(directed=self.directed)
		self.setup()
		self.debug2 = debug2
		self.max_tries = max_tries
		if self.debug2: print "setup done"

	def get_length(self):
		if random.random() < self.cc: return 1
		return 2

	def setup(self):
		if not self.gpre:
			self.gpre = ig.Graph.Erdos_Renyi(self.n0, m=self.n0*self.m, directed=self.directed)

		self.use_chunks = False
		self.n0 = len(self.gpre.vs)
		self.total_edges = len(self.gpre.es)
		self.next_nid = self.n0
		self.chunk_nid = self.next_nid-1
		self.chunk_size = 1

		self.nbors = defaultdict(list)
		self.out_nbors = defaultdict(list)
		self.in_nbors = defaultdict(list)
		self.nid_chunk_map = {}

		for nid, nbor_nids in enumerate(self.gpre.get_adjlist(mode='ALL')): self.nbors[nid] = nbor_nids
		for nid, nbor_nids in enumerate(self.gpre.get_adjlist(mode='OUT')): self.out_nbors[nid] = nbor_nids
		for nid, nbor_nids in enumerate(self.gpre.get_adjlist(mode='IN')): self.in_nbors[nid] = nbor_nids
		for nid in self.gpre.vs.indices: self.nid_chunk_map[nid] = 0

		# node length attribute
		self.nid_length_map = {nid: self.get_length() for nid in xrange(self.next_nid)}

	def add_nodes(self, chunk_seq, mean_seq):
		num_chunks = len(chunk_seq)
		chunk_debug = num_chunks//10
		if (self.debug): print "Total chunks: {}".format(num_chunks)

		for idx, (chunk_size, m) in enumerate(zip(chunk_seq, mean_seq), 1):
			if self.debug2: print "Adding {} nodes with outdeg {:.2f}".format(chunk_size, m)
			if self.debug and idx % chunk_debug == 0: print idx,
			self.chunk_size = chunk_size
			self.m = m
			self.add_chunk(idx)
			self.chunk_nid = self.next_nid-1

		self.build_graph()

	def add_chunk(self, chunk_id):
		marked = defaultdict(set)

		for _ in xrange(self.chunk_size):
			new_nid = self.next_nid; self.next_nid += 1
			self.nid_chunk_map[new_nid] = chunk_id
			marked[new_nid] = self.add_node(new_nid)

		self.update_chunk(marked)

	def build_graph(self):
		self.edges = edges = set()
		all_nbors = self.out_nbors if self.directed else self.nbors

		for node, nbors in all_nbors.iteritems():
			for nbor in nbors:
				if self.directed: edges.add((node, nbor))
				else: edges.add((max(node, nbor), min(node, nbor)))

		self.g.add_vertices(self.next_nid)
		self.g.add_edges(list(edges))
		self.g.simplify()
		self.g.vs['chunk_id'] = [self.nid_chunk_map[n] for n in self.g.vs.indices]

		if self.debug: print "\n{}".format(self.g.summary())

	def update_chunk(self, marked_dict):
		for nid, marked in marked_dict.items():
			self.update_node(nid, marked)

	def update_node(self, nid, marked):
		for nbor_nid in marked:
			self.nbors[nid].append(nbor_nid)
			self.nbors[nbor_nid].append(nid)
			self.out_nbors[nid].append(nbor_nid)
			self.in_nbors[nbor_nid].append(nid)

	def get_seed_nid(self, new_nid, size=1):
		max_nid = new_nid-1
		min_nid = 0
		if size == 1: return np.random.randint(min_nid, max_nid+1)
		return np.random.randint(min_nid, max_nid+1, size)

	def _walk(self, cur_nid, length):
		for _ in xrange(length):
			nbors = self.nbors[cur_nid]
			if not nbors: return -1
			cur_nid = random.choice(nbors)
		return cur_nid

	def walk(self, cur_nid, length):
		next_nid = self._walk(cur_nid, length)
		return next_nid

	def _get_cur_and_seed_nid(self, new_nid):
		cur_nid = -1
		while cur_nid == -1:
			seed_nid = self.get_seed_nid(new_nid)
			cur_nid = self.walk(seed_nid, 2)
		return cur_nid, seed_nid

	def add_node(self, new_nid):
		marked = set()
		m = min(len(self.nbors), self.m)
		m_ = int(m)
		if random.random() < m-m_: m += 1

		self.nid_length_map[new_nid] = self.get_length()
		if m == 0: return marked

		cur_nid, seed_nid = self._get_cur_and_seed_nid(new_nid)
		marked.add(cur_nid)

		tries = 0
		while len(marked) < m:
			prev_nid = cur_nid
			cur_nid = self.walk(cur_nid, self.nid_length_map[cur_nid])
			if cur_nid not in marked: marked.add(cur_nid)
			else:
				tries += 1
				if tries % self.max_tries == 0:
					# stuck, restart
					cur_nid, seed_nid = self._get_cur_and_seed_nid(new_nid)
					marked.add(cur_nid)
		return marked

class SaramakiKaskiRandomWalk(HereraZufiriaRandomWalk):

	def __init__(self, l, gpre=None, directed=True, debug=True, debug2=False, er_m=5, er_n0=50, max_tries=100):
		self.l = l
		super(SaramakiKaskiRandomWalk, self).__init__(-1, gpre=gpre, directed=directed, debug=debug, debug2=debug2, er_m=er_m, er_n0=er_n0, max_tries=max_tries)

	def _walk(self, cur_nid, length):
		for _ in xrange(length):
			nbors = self.nbors[cur_nid]
			if not nbors: return cur_nid
			cur_nid = random.choice(nbors)
		return cur_nid

	def walk(self, cur_nid, length):
		next_nid = self._walk(cur_nid, length)
		return next_nid

	def add_node(self, new_nid):
		marked, m = set(), min(len(self.nbors), self.m)
		m_ = int(m)
		if random.random() < m-m_: m += 1
		if m == 0: return marked

		seed_nid = cur_nid = self.get_seed_nid(new_nid)
		tries = 0

		while len(marked) < m:
			cur_nid = self.walk(cur_nid, self.l)

			if cur_nid in marked:
				tries += 1
				if tries > self.max_tries: cur_nid = self.get_seed_nid(new_nid)
			else:
				marked.add(cur_nid)
				tries = 0 # reset

		return marked