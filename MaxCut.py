import cvxpy as cvx
import numpy as np

def maxcut_random(G, is_adj_list=True):
	# calculates the maximum weight cut by randomly assigning vertices to a cut. This is
	# a 1/2-approximation algorithm in expectation.
	#
	# input:
	#	G: a graph in adjacency list format, and weights for each edge. Format is a list
	#	of |V| lists, where each internal list is either a list of integers in {0, |V|-1}
	#	or a list of 2-tuples, each contaning an integer in {0, |V|-1} and a positive
	#	real weight. No multi-edges or directed edges are allowed, so internal lists are 
	#	at most |V|-1 long. However, this (and non-negative weights) is not enforced.
	#
	# output:
	#	chi: a list of length |V| where the ith element is +1 or -1, representing which
	#	set the ith vertex is in. Returns None if an error occurs.
	if len(G) < 1:
		return None
	if len(G) == 1:
		return [1]
	# return [1 if np.random.sample() >= 0.5 else -1 for i in range(len(G))]
	chi = np.zeros(len(G))
	resevoir = [1 if elem < len(G) // 2 else -1 for elem in range(len(G))]
	for i in range((len(G))):
		chi[i] = resevoir.pop(np.random.randint(0, len(resevoir)))
	return chi

def maxcut_greedy(G, is_adj_list=True):
	# calculates the maximum weight cut using a simple greedy implementation. This is a
	# 1/2-approximation algorithm.
	#
	# input:
	#	G: a graph in adjacency list format, and weights for each edge. Format is a list
	#	of |V| lists, where each internal list is either a list of integers in {0, |V|-1}
	#	or a list of 2-tuples, each contaning an integer in {0, |V|-1} and a positive
	#	real weight. No multi-edges or directed edges are allowed, so internal lists are 
	#	at most |V|-1 long. However, this (and non-negative weights) is not enforced.
	#
	# output:
	#	chi: a list of length |V| where the ith element is +1 or -1, representing which
	#	set the ith vertex is in. Returns None if an error occurs.
	
	l, r, V, l_cost, r_cost = set(), set(), len(G), 0, 0
	if is_adj_list:
		for num, v in enumerate(G):
			for u in v:
				if isinstance(u, tuple):
					if u[0] in l:
						l_cost += u[1]
					if u[0] in r:
						r_cost += u[1]
				else:
					if u in l:
						l_cost += 1
					if u in r:
						r_cost += 1
			if l_cost <= r_cost: # r_cost is amount by which cut will increase
				l.add(num)
			else:
				r.add(num)
			l_cost, r_cost = 0, 0
	else:
		for i in range(V):
			for j in range(V):
				if j in l:
					l_cost += G[i, j]
				if j in r:
					r_cost += G[i, j]
			if l_cost <= r_cost:
				l.add(i)
			else:
				r.add(i)
			l_cost, r_cost = 0, 0
	return [1 if i in l else -1 for i in range(V)]

def maxcut_SDP(G, is_adj_list=True):
	# calculates the maximum weight cut by generating |V| vectors with a vector program,
	# then generating a random plane that cuts the vertices. This is a .878-approximation
	# algorithm.
	#
	# input:
	#	G: a graph in adjacency list format, and weights for each edge. Format is a list
	#	of |V| lists, where each internal list is either a list of integers in {0, |V|-1}
	#	or a list of 2-tuples, each contaning an integer in {0, |V|-1} and a positive
	#	real weight. No multi-edges or directed edges are allowed, so internal lists are 
	#	at most |V|-1 long. However, this (and non-negative weights) is not enforced.
	#
	# output:
	#	chi: a list of length |V| where the ith element is +1 or -1, representing which
	#	set the ith vertex is in. Returns None if an error occurs.
	# setup
	V, x, constraints, expr = len(G), [], [], 0
	if is_adj_list:
		G = _adj_list_to_adj_matrx(G)

	# variables				
	X = cvx.Variable((V, V), PSD=True)

	# constraints
	for i in range(V):
		constraints.append(X[i, i] == 1)

	# objective function	
	expr = cvx.sum(cvx.multiply(G, (np.ones((V, V)) - X)))

	# solve
	prob = cvx.Problem(cvx.Maximize(expr), constraints)
	prob.solve()

	# random hyperplane partitions vertices
	Xnew = X.value
	eigs = np.linalg.eigh(Xnew)[0]
	if min(eigs) < 0:
		Xnew = Xnew + (1.00001 * abs(min(eigs)) * np.identity(V))
	elif min(eigs) == 0:
		Xnew = Xnew + 0.0000001 * np.identity(V)
	x = np.linalg.cholesky(Xnew).T
	r = np.random.normal(size=(V))
	return [1 if np.dot(x[i], r) >= 0 else -1 for i in range(V)]

def _adj_matrix_to_adj_list(mat):
	# converts an np.ndarray of shape (|V|, |V|) of non-negative real values to a graph 
	# adjacency list. 
	#
	# input:
	#	mat: a list of |V| lists, each containing ints in {0, |V|-1} or 2-tuples of an
	# 	int in {0, |V|-1} and a positive real weight.
	#
	# output:
	#	G: np.ndarray of shape (|V|, |V|)
	return None

def _adj_list_to_adj_matrx(lst):
	# converts an adjacency list np.ndarray of shape (|V|, |V|) of non-negative real 
	# values.
	#
	# input:
	#	lst: np.ndarray of shape (|V|, |V|)
	#
	# output:
	#	G: a list of |V| lists, each containing ints in {0, |V|-1} or 2-tuples of an
	# 	int in {0, |V|-1} and a positive real weight.
	V, weighted = len(lst), False
	for i in range(V):
		if len(lst[i]) > 0:
			if isinstance(lst[i][0], tuple):
				weighted = True
				break
	weighted_adj_matrix = np.zeros((V, V))
	for i in range(V):
		for j in range(len(lst[i])):
			if weighted:
				u, v = i, lst[i][j][1]
				weighted_adj_matrix[u, v] = lst[i][j][0]
				weighted_adj_matrix[v, u] = lst[i][j][0]
			else:
				u, v = i, lst[i][j]
				weighted_adj_matrix[u, v] = 1
				weighted_adj_matrix[v, u] = 1
	return weighted_adj_matrix

def _gnp_random_graph_adj_matrix(n, prob, weighted=False):
	# generates a Gnp random graph: a graph with n vertices where each edge occurs with
	# probability p
	#
	# input:
	#	n: the number of vertices in the graph
	#	p: the probability of each edge existing
	# 
	# output:
	#	A: a numpy array of dimensions n by n representing the adjacency matrix
	A = np.zeros((n, n))
	if not weighted:
		for i in range(n):
			for j in range(i + 1, n):
				sample = np.random.choice((0, 1), p=[1 - prob, prob])
				A[i, j] = sample
				A[j, i] = sample
	else:
		for i in range(n):
			for j in range(i + 1, n):
				if np.random.random_sample() >= 1 - prob:
					sample = max(np.random.normal(1, .5), 0)
					A[i, j] = sample
					A[j, i] = sample
	return A

def _eval_cut(G, chi):
	# calculates total weight across a cut
	#
	# input:
	#	G: a numpy array representing an adjacency matrix
	#	chi: an array where all elements are +1 or -1, representing which side of the cut
	#	that vertex is in.
	#
	#
	total, V = 0, G.shape[0]
	for i in range(V):
		for j in range(i + 1, V):
			if chi[i] != chi[j]:
				total += G[i, j]
	return total

def _run_tests():
	rand_total, greedy_total, SDP_total, ratio = 0, 0, 0, []
	i = 2
	while i <= 128:
		for j in range(100):
			G = _gnp_random_graph_adj_matrix(i, 0.5, True)
			rand = _eval_cut(G, maxcut_random(G, False))
			rand_total += rand
			# print("random for instance " + str(j) + ": " + str(rand))
			greedy = _eval_cut(G, maxcut_greedy(G, False))
			greedy_total += greedy
			# print("greedy for instance " + str(j) + ": " + str(greedy))
			SDP = _eval_cut(G, maxcut_SDP(G, False))
			SDP_total += SDP
			# print("SDP for instance " + str(j) + ": " + str(SDP))
		i *= 2
		print("final scores:")
		print("random: " + str(rand_total / SDP_total))
		print("greedy: " + str(greedy_total / SDP_total))
		print("SDP to greedy: " + str(SDP_total / greedy_total))
		ratio.append(SDP_total / greedy_total)
	print(ratio)

_run_tests()