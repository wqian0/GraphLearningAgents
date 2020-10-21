import copy
from colour import Color
import mynumpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression

from matplotlib import colors
from operator import itemgetter
from graphviz import Digraph
import os
import pickle as pk
from copy import deepcopy
import copy
from depq import DEPQ
import pandas as pd
import networkx as nx
import scipy as sp
from scipy import optimize as op
import random as rd
import msvcrt
from scipy import stats
os.environ["PATH"] += os.pathsep + 'C:/Users/billy/Downloads/graphviz-2.38/release/bin'
import small_world as sw
import igraph as ig
import MidpointNormalize as mn
import unionfind as uf
import copyreg
import types
import time

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.seterr(divide='ignore')

rng = np.random.RandomState()
#seeded_rng = np.random.RandomState(17310145)
seeded_rng = rng

N_internal = 24

head_dir = "C:/Users/billy/PycharmProjects/GraphLearningAgents/"

languages = head_dir + "graphs_Language_share/"
music = head_dir + "graphs_Music_share/"
web = head_dir + "graphs_Web_share/"
social = head_dir + "graphs_Social_share/"
citation = head_dir + "graphs_Citation_share/"
semantic = head_dir + "graphs_Semantic_share/"

def get_random_modular(n, modules, edges, p, getCommInfo=False):
    pairings = {}
    assignments = np.zeros(n, dtype = int)
    cross_module_edges = []
    for i in range(modules):
        pairings[i] = []
    A = np.zeros((n,n))
    for i in range(n):
        randomModule = seeded_rng.randint(0, modules)
        pairings[randomModule].append(i)
        assignments[i] = randomModule
    for i in range(modules - 1):
        if len(pairings[i]) < 3 or len(pairings[i+1]) < 3:
            return None, None
        e0, e1 = seeded_rng.choice(pairings[i], 1), seeded_rng.choice(pairings[i+1], 1)
        A[e0, e1], A[e1, e0] = 1, 1
        cross_module_edges.append((e0, e1))
    def add_modular_edge():
        randomComm = seeded_rng.randint(0, modules)
        while len(pairings[randomComm]) < 2:
            randomComm = seeded_rng.randint(0, modules)
        selection = seeded_rng.choice(pairings[randomComm], 2, replace=False)
        while A[selection[0], selection[1]] != 0:
            randomComm = seeded_rng.randint(0, modules)
            while len(pairings[randomComm]) < 2:
                randomComm = seeded_rng.randint(0, modules)
            selection = seeded_rng.choice(pairings[randomComm], 2, replace=False)
        A[selection[0], selection[1]] += 1
        A[selection[1], selection[0]] += 1

    def add_between_edge():
        randEdge = seeded_rng.choice(n, 2, replace=False)
        while A[randEdge[0], randEdge[1]] != 0 or assignments[randEdge[0]] == assignments[randEdge[1]]:
            randEdge = seeded_rng.choice(n, 2, replace=False)
        A[randEdge[0], randEdge[1]] += 1
        A[randEdge[1], randEdge[0]] += 1
        cross_module_edges.append(randEdge)
    inModuleEdges = int(round(edges * p))
    betweenEdges = edges - inModuleEdges - modules + 1
    if betweenEdges < 0:
        print("RIP NEGATIVE")
    for i in range(inModuleEdges):
        add_modular_edge()
    for i in range(betweenEdges):
        add_between_edge()
    def parameterized(cc_weight):
        B = deepcopy(A)
        for e in cross_module_edges:
            B[e[0], e[1]], B[e[1], e[0]] = cc_weight, cc_weight
        return B
    if getCommInfo:
        return A, pairings, assignments
    else:
        return A, parameterized

def learn(A, beta):
    A = normalize(A)
    inverse_argument = np.identity(len(A)) - np.exp(-beta)*A
    inverse = sp.linalg.inv(inverse_argument)
    return normalize((1-np.exp(-beta))*(A @ inverse))

def learn_nonormalize(A, beta):
    inverse_argument = np.identity(len(A)) - np.exp(-beta)*A
    inverse = sp.linalg.inv(inverse_argument)
    return (1-np.exp(-beta))*(A @ inverse)

def get_stationary(A):
    lam, vec = sp.linalg.eig(A, left=True, right=False)
    idx = np.argmin(np.abs(lam - 1))
    w = np.real(vec[:, idx])
    return w / w.sum().real

def get_stationary2(A):
    P = np.linalg.matrix_power(A, 10000)
    P_next = np.dot(P, A)
    while not np.allclose(P, P_next):
        P_next = np.dot(P_next, A)
    return P_next[0]

def normalize(A, delLoops = False):
    B = deepcopy(A)
    J = np.ones((len(A), len(A)))
    output = B / (B @ J)
    output[np.isnan(output)] = 0
    return output

def unnormalize(A):
    pi = get_stationary2(A)
    return np.einsum('i, ij -> ij', pi, A)
#Assumes undirected input!

def getNumEdges(A):
    return np.count_nonzero(A) / 2

#only applies for unnormalized weighted graph inputs
def get_stationary3(A):
    output = np.sum(A, axis = 0) / (np.sum(A))
    return output

def KL_Div_Old(U, V):
    U = normalize(U)
    V = normalize(V)
    pi = get_stationary2(U)
    result = 0
    for i in range(len(U)):
        for j in range(len(U)):
            if not np.isclose(U[i][j], 0, rtol = 1e-16) and not np.isclose(V[i][j], 0, rtol = 1e-16):
                result += pi[i] * U[i][j] * np.log(V[i][j]/U[i][j])
    return -result

def KL_Divergence(U, V, weighted_net = None):
    U = normalize(U)
    V = normalize(V)
    if weighted_net is None:
        pi = get_stationary2(U)
    else:
        pi = get_stationary3(weighted_net)
    combined = np.einsum('i, ij -> ij', pi, U)
    logged = np.log(V/U)
    logged[U == 0] = 0
    # outcome = -np.einsum('ij, ji ->', combined.T, logged)
    result = combined.T @ logged
    outcome = -np.trace(result)
    return outcome

def create_network(N, edges):
    adjMatrix = np.zeros((N, N), dtype = float)
    for i in range(edges):
        randEdge = seeded_rng.choice(N, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = seeded_rng.choice(N, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1.0
    return adjMatrix

def create_undirected_network(N, edges):
    adjMatrix = np.zeros((N, N), dtype = float)
    perm = np.arange(N)
    for i in range(N - 1):
        adjMatrix[perm[i]][perm[i+1]] = 1.0
        adjMatrix[perm[i+1]][perm[i]] = 1.0
    for i in range(edges - N + 1):
        randEdge = seeded_rng.choice(N, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = seeded_rng.choice(N, 2, replace=False)
            #print("STUCK")
        adjMatrix[randEdge[0]][randEdge[1]] = 1.0
        adjMatrix[randEdge[1]][randEdge[0]] = 1.0
    return adjMatrix

def get_laplacian(A):
    result = -deepcopy(A)
    for i in range(len(result[0])):
        result[i][i] += np.sum(A[i])
    return result

def isConnected(A, returnInfo = False):
    L = get_laplacian(A)
    evals, v = sp.linalg.eigh(L, eigvals=(1, len(A) - 1))
    if returnInfo:
        return evals[0] > 1e-10, evals
    return evals[0] > 1e-10

def create_modular_toy(edges, modular_edges):
    adjMatrix = np.zeros((15, 15))
    module_1, module_2, module_3 = np.arange(5), np.arange(5, 10), np.arange(10, 15)
    modules = []
    modules.append(module_1), modules.append(module_2), modules.append(module_3)
    def add_modular_edge():
        randomComm = seeded_rng.randint(0, 3)
        randEdge = seeded_rng.choice(modules[randomComm], 2, replace = False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0 or adjMatrix[randEdge[1]][randEdge[0]] != 0:
            randomComm = seeded_rng.randint(0, 3)
            randEdge = seeded_rng.choice(modules[randomComm], 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
        adjMatrix[randEdge[1]][randEdge[0]] += 1

    def add_cross_edge(): #adds edge outside modules
        randEdge = seeded_rng.choice(15, 2, replace = False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0 or adjMatrix[randEdge[1]][randEdge[0]] != 0 or \
        (randEdge[0] in module_1 and randEdge[1] in module_1) \
                or (randEdge[0] in module_2 and randEdge[1] in module_2) or \
                (randEdge[0] in module_3 and randEdge[1] in module_3):
            randEdge = seeded_rng.choice(15, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
        adjMatrix[randEdge[1]][randEdge[0]] += 1
    for i in range(modular_edges):
        add_modular_edge()
    for i in range(edges - modular_edges):
        add_cross_edge()
    return adjMatrix

def modular_toy_paper(): ##constructs the three-community network, often used in studying human information processing
    result = np.zeros((15, 15))
    for i in range(5):
        for j in range(5):
            result[i][j] += 1.0
    for i in range(5, 10):
        for j in range(5, 10):
            result[i][j] += 1.0
    for i in range(10, 15):
        for j in range(10, 15):
            result[i][j] += 1.0
    for i in range(15):
        result[i][i] = 0

    result[0][4], result[4][0] = 0, 0
    result[0][14], result[14][0] = 1, 1

    result[5][9], result[9][5] = 0, 0
    result[4][5], result[5][4] = 1, 1

    result[9][10], result[10][9] = 1, 1
    result[10][14], result[14][10] =0, 0
    return result

def biased_modular(cross_cluster_bias, boundary_bias):
    result = modular_toy_paper()
    result[0][14], result[14][0] = cross_cluster_bias, cross_cluster_bias
    result[4][5], result[5][4] = cross_cluster_bias, cross_cluster_bias
    result[9][10], result[10][9] = cross_cluster_bias, cross_cluster_bias
    for i in [0, 4]:
        for j in range(1, 4):
            result[i][j], result[j][i] = boundary_bias, boundary_bias

    for i in [5, 9]:
        for j in range(6, 9):
            result[i][j], result[j][i] = boundary_bias, boundary_bias

    for i in [10, 14]:
        for j in range(11, 14):
            result[i][j], result[j][i] = boundary_bias, boundary_bias
    return result

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def flipEdge(A, ensureConnected = True):
    B = deepcopy(A)
    edges = []
    empty = []
    for r in range(len(A) - 1):
        for c in range(r + 1, len(A)):
            if B[r][c] != 0.0 and B[r][c] != 0:
                edges.append((r, c))
            else:
                empty.append((r, c))
    rE = seeded_rng.randint(len(edges))
    rA = seeded_rng .randint(len(empty))
    B[edges[rE][0]][edges[rE][1]] = 0.0
    B[edges[rE][1]][edges[rE][0]] = 0.0
    B[empty[rA][0]][empty[rA][1]] = 1.0
    B[empty[rA][1]][empty[rA][0]] = 1.0
    if ensureConnected:
        if not isConnected(B):
            B = flipEdge(A)
        else:
            return B
    return B

def find_edgepair(A, e0, e1):
    newEdges = []
    if A[e0[0]][e1[0]] == 0.0 and A[e0[1]][e1[1]] == 0.0 and e0[0] != e1[0] and e0[1] != e1[1]:
        newEdges.append([(e0[0], e1[0]), (e0[1], e1[1])])
    if A[e0[1]][e1[0]] == 0.0 and A[e0[0]][e1[1]] == 0.0 and e0[1] != e1[0] and e0[0] != e1[1]:
        newEdges.append([(e0[1], e1[0]), (e0[0], e1[1])])
    if len(newEdges) == 0:
        return None
    else:
        return newEdges[seeded_rng.choice(len(newEdges))]

def rewire_regular(A, ensureConnected = True):
    B = deepcopy(A)
    newEdges = None
    dels, adj0, adj1 = 0,0,0
    while newEdges == None or adj0 == adj1:
        dels = seeded_rng.choice(len(B), 2, replace=False)
        adj0 = seeded_rng.choice(np.nonzero(B[dels[0]])[0])
        adj1 = seeded_rng.choice(np.nonzero(B[dels[1]])[0])
        newEdges = find_edgepair(B, (dels[0], adj0), (dels[1], adj1))

    B[dels[0]][adj0] = 0
    B[adj0][dels[0]] = 0
    B[dels[1]][adj1] = 0
    B[adj1][dels[1]] = 0
    B[newEdges[0][0]][newEdges[0][1]] = 1
    B[newEdges[0][1]][newEdges[0][0]] = 1
    B[newEdges[1][0]][newEdges[1][1]] = 1
    B[newEdges[1][1]][newEdges[1][0]] = 1
    if ensureConnected:
        if not isConnected(B):
            B = rewire_regular(A, ensureConnected = True)
        else:
            return B
    return B

def KL_score(A, beta, A_target = None):
    return KL_Divergence(A, learn(A, beta))

def KL_score_ext_zipped(input, beta):
    cc_bias, b_bias = input
    if cc_bias < 1e-8 or b_bias < 1e-8:
        return 100
    return KL_score_external(biased_modular(cc_bias, b_bias), beta, modular_toy_paper())

def KL_score_external(A_input, beta, A_target, weighted_net = None):
    return KL_Divergence(A_target, learn(A_input, beta), weighted_net = weighted_net)

def automorphism_count(A, beta, A_target = None):
    IG = ig.Graph.Adjacency(A.tolist())
    return IG.count_automorphisms_vf2()

def get_regular_graph(N, d):
    return np.array(nx.to_numpy_matrix(nx.random_regular_graph(d, N)))

def get_lattice_graph(dim):
    return np.array(nx.to_numpy_matrix(nx.grid_graph(dim, periodic = True)))

def get_automorphisms(A):
    IG = ig.Graph.Adjacency(A.tolist())
    return np.transpose(np.array(IG.get_automorphisms_vf2()))

def get_structurally_symmetric(A, force_unique = False):
    disj_set = uf.UnionFind()
    for i in range(len(A)):
        disj_set.add(i)
    if force_unique:
        return disj_set.components(), disj_set.component_mapping()
    autos = get_automorphisms(A)
    for i in range(len(A)):
        #print(i)
        for j in range(len(autos[0])):
            disj_set.union(i, autos[i][j])
    return disj_set.components(), disj_set.component_mapping()

#input: symmetric adj matrix
def compute_line_graph(A):
    G_input = nx.from_numpy_matrix(A)
    return nx.to_numpy_matrix(nx.line_graph(G_input))

def unique_edges(A, beta, A_target = None):
    edge_labels, inv_labels, line_graph = compute_line_graph_details(A)
    components, comp_mappings = get_structurally_symmetric(line_graph)
    return components, comp_mappings, edge_labels, inv_labels

def compute_line_graph_details(A):
    edge_labels = dict()
    count = 0
    for r in range(len(A)-1):
        for c in range(r+1, len(A)):
            if A[r][c] == 1.0:
                edge_labels[(r,c)] = count
                count += 1
    inv_labels = {v: k for k, v in edge_labels.items()}
    keys = list(edge_labels.keys())
    edges = int(np.sum(A) / 2)
    for k in keys:
        r,c = k
        edge_labels[(c,r)] = edge_labels[(r,c)]
    output = np.zeros((edges, edges))
    for i in range(edges - 1):
        a, b = inv_labels[i]
        for j in range(i + 1, edges):
            c, d = inv_labels[j]
            if a == c or a == d or b == c or b == d:
                if not ((a == d and b == c) or (a == c and b == d)):
                    output[i][j], output[j][i] = 1, 1
    return edge_labels, inv_labels, output

def get_complement_graph(A):
    B = np.zeros((len(A), len(A)))
    B[A == 0] = 1
    B[A == 1] = 0
    for i in range(len(A)):
        B[i][i] = 0
    return B

def getSymReducedParams(A, include_nonexistent = True):
    comps, comp_maps, edge_labels, inv_labels = unique_edges(A, 0)
    if include_nonexistent:
        A_c = get_complement_graph(A)
        comps_c, comp_maps_c, edge_labels_c, inv_labels_c = unique_edges(A_c, 0)
    def parameterized(input):
        B = np.zeros((len(A), len(A)))
        for i in range(len(comps)):
            for x in comps[i]:
                row, col = inv_labels[x]
                B[row][col], B[col][row] = input[i], input[i]
        if include_nonexistent:
            for i in range(len(comps_c)):
                for x in comps_c[i]:
                    row, col = inv_labels_c[x]
                    B[row][col], B[col][row] = input[len(comps) + i], input[len(comps)+i]
        return B
    if include_nonexistent:
        return len(comps) + len(comps_c), parameterized
    return len(comps), parameterized


def get_KL_ext_general(A_target, include_nonexistent = True):
    numParams, parameterized = getSymReducedParams(A_target, include_nonexistent = include_nonexistent)
    def cost_func(input, P_0, pi, beta, J, I):
        return KL_score_external(parameterized(input), beta, A_target)
    return numParams, cost_func, parameterized

def get_KL_ext_mod(parameterized):
    def cost(cc_weight, beta, A_target):
        return KL_score_external(parameterized(cc_weight), beta, A_target)
    return cost

def get_pickleable_params(A, include_nonexistent = True):
    comps, comp_maps, edge_labels, inv_labels = unique_edges(A, 0)
    if include_nonexistent:
        A_c = get_complement_graph(A)
        comps_c, comp_maps_c, edge_labels_c, inv_labels_c = unique_edges(A_c, 0)
        return len(comps) + len(comps_c), comps, comps_c, inv_labels, inv_labels_c
    return len(comps), comps, None, inv_labels, None

def pickleable_cost_func(input, comps, comps_c, inv_labels, inv_labels_c, beta, A_target, include_nonexistent,
                         KL = True, weighted_net = None):
    B = np.zeros((len(A_target), len(A_target)))
    for i in range(len(comps)):
        for x in comps[i]:
            row, col = inv_labels[x]
            B[row][col], B[col][row] = input[i], input[i]
    if include_nonexistent:
        for i in range(len(comps_c)):
            for x in comps_c[i]:
                row, col = inv_labels_c[x]
                B[row][col], B[col][row] = input[len(comps) + i], input[len(comps) + i]
    if KL:
        return KL_score_external(B, beta, A_target, weighted_net = weighted_net)
    else:
        return uniformity_cost(A_target, B, beta)


def reduced_cost_func(input, comps, comps_c, inv_labels, inv_labels_c, beta, A_target, indices_c):
    B = np.zeros((len(A_target), len(A_target)))
    for i in range(len(comps)):
        for x in comps[i]:
            row, col = inv_labels[x]
            B[row][col], B[col][row] = input[i], input[i]
    for i in range(len(indices_c)):
        for x in comps_c[indices_c[i]]:
            row, col = inv_labels_c[x]
            B[row][col], B[col][row] = input[len(comps) + i], input[len(comps) + i]
    return KL_score_external(B, beta, A_target)

def grad_zipped(input, P_0, pi, beta, J, I):
    iu = np.triu_indices(len(P_0), k=1)
    A = np.zeros((len(P_0), len(P_0)))
    eta = np.exp(-beta)
    A[iu] = input
    A = np.maximum(A, A.T)
    output = gradient(P_0, pi, A, eta, J, I)
    return output[iu]

def cost_func_zipped(input, P_0, pi, beta, J, I):
    iu = np.triu_indices(len(P_0), k=1)
    A = np.zeros((len(P_0), len(P_0)))
    A[iu] = input
    A = np.maximum(A, A.T)
    cost = cost_func(P_0, pi, A, beta, J, I)
    print(cost)
    return cost

def Q_inv(A, beta, depth):
    nu = np.exp(-beta)
    result = 0
    for i in range(depth):
        result += np.linalg.matrix_power(nu * A, i)
    return result

def cost_func(P_0, pi, A, beta, J, I):
    AJ = A @ J
    P_f = A / AJ
    P_f[np.isnan(P_f)] = 0
    eta = np.exp(-beta)
    Q = sp.linalg.inv(I - eta * P_f)
    prod = P_f @ Q
    M2 = np.log(prod/ P_0)
    M2[P_0 == 0] = 0
    combined = np.einsum('i, ij -> ij', pi, P_0)
    return -np.log(1-eta)-np.trace(combined.T @ M2)

def gradient(P_0, pi, A, eta, J, I):
    AJ = A @ J
    P_f = A / AJ
    P_f[np.isnan(P_f)] = 0
    Q = sp.linalg.inv(I- eta * P_f)
    combined = np.einsum('i, ij -> ij', pi, P_0)
    R = combined / (P_f @ Q)
    R[np.isnan(R)] = 0
    S = (eta * Q.T @ P_f.T - I) @ (R @ Q.T)/ (AJ * AJ)
    S[np.isnan(S)] = 0
    output = S * AJ - (S*A) @ J
    return output

def descend(P_0, pi, beta, rate, iterations):
    eta = np.exp(-beta)
    I = np.identity(len(P_0))
    A = biased_modular(.25, .8)
    J = np.ones((len(P_0), len(P_0)))
    count = 0
    norm = np.inf
    mod = modular_toy_paper()
    while count < iterations or norm > 1e-8:
        descent = rate / (len(P_0)** 2) * gradient(P_0, pi, A, eta, J, I)
        A -= descent
        count += 1
        norm = sp.linalg.norm(descent)
        A[A < 0] = 0
        if count % 1000 == 0:
            cost = cost_func(P_0, pi, A, eta, J, I)
            print(str(norm)+"\t"+str(count)+"\t"+str(np.min(A))+"\t"+
                  str(KL_score_external(A, .3, modular_toy_paper()))+"\t"+str(cost))
    result = A / np.dot(A, J)
    return result

def compose(f, n):
    def fn(x):
        for _ in range(n):
            x = f(x)
        return x
    return fn

def transitivity_score(A, beta):
    G = nx.from_numpy_matrix(A)
    return nx.transitivity(G)

def symmetry_toy():
    A = np.zeros((15, 15))
    for i in range(5):
        for j in range(i + 1, 6):
            A[i][j], A[j][i] = 1, 1
    for i in range(8, 15):
        A[i][6], A[6][i] = 1, 1
        A[i][7], A[7][i] = 1, 1
    A[5][6], A[6][5] = 1, 1
    return A

def optimize(A, beta, iterations, scoreFunc, flipFunc, minimize = True, A_target = None, numParams = False):
    bestVal = float('inf')
    curr = deepcopy(A)
    best = deepcopy(A)
    factor = 1
    if not minimize:
        factor = -1
        bestVal = -float('inf')
    for i in range(iterations):
        if numParams:
            paramCount, parameterized = scoreFunc(curr)
            score = paramCount
        else:
            score = scoreFunc(curr, beta, A_target)
        currScore = factor * score
        if currScore <= bestVal and isConnected(curr):
            bestVal = currScore
            best = deepcopy(curr)
        curr = compose(flipFunc, seeded_rng.randint(15))(best)
        count = 0
        while not isConnected(curr):
            curr = compose(flipFunc, seeded_rng.randint(15))(best)
            count += 1
            if count > 30:
                return bestVal, best
        print(str(i)+"\t"+str(currScore)+"\t"+str(bestVal))
    return bestVal,  best

def uniformity_cost(P_0, A, beta):
    learned = learn(A, beta)
    terms = learned[P_0 > 0].flatten()
    diffs = np.subtract.outer(terms, terms)
    return np.sum(diffs * diffs)

def uniformity_cost_zipped(input, N_tot, N_comms, beta):
    A = modular_toys_general(N_tot, N_comms, input[0], input[1])
    return uniformity_cost(modular_toys_general(N_tot, N_comms, 1, 1), A, beta)

def modular_toys_general(N_tot, N_comms, cc_bias, b_bias):
    A = np.zeros((N_tot, N_tot))
    N_in = N_tot // N_comms
    b = [] #boundary nodes
    for i in range(N_comms):
        for j in range(i * N_in, (i + 1) * N_in - 1):
            for k in range(j + 1, (i + 1) * N_in):
                A[j][k], A[k][j] = 1.0, 1.0
        A[i * N_in][(i + 1) * N_in - 1], A[(i + 1) * N_in - 1][i * N_in] = 0, 0

    for i in range(N_comms):
        b.append(i * N_in)
        b.append((i + 1) * N_in - 1)
        for j in [i * N_in, (i + 1) * N_in - 1]:
            for k in range(i * N_in + 1, (i + 1) * N_in - 1):
                A[j][k], A[k][j] = b_bias, b_bias
    for i in range(1, len(b) // 2):
        A[b[2*i - 1]][b[2*i]], A[b[2*i]][b[2*i - 1]] = cc_bias, cc_bias
    A[b[0]][b[len(b) - 1]], A[b[len(b) - 1]][b[0]] = cc_bias, cc_bias
    return A

def KL_modular_toys_general(input, N_tot, N_comms, beta):
    cc_bias, b_bias = input
    return KL_score_external(modular_toys_general(N_tot, N_comms, cc_bias, b_bias), beta, modular_toys_general(N_tot, N_comms, 1, 1))

def frobenius_norm_cost(P_0, A, beta):
    learned = learn(A, beta)
    return sp.linalg.norm(normalize(P_0) - learned)

def list_optimize(A_list, beta, iterations, scoreFunc, flipFunc, listSize, minimize = True, A_target = None, numParams = False):
    factor = -1
    deque = DEPQ(iterable = None, maxlen = listSize)
    if not minimize:
        factor = 1
    for i in range(listSize):
        deque.insert(deepcopy(A_list[i]), -105)
    for i in range(iterations):
        for j in range(listSize):
            element = deque[j]
            curr = compose(flipFunc, seeded_rng.randint(len(A_list[0])))(element[0])
            count = 0
            while not isConnected(curr):
                curr = compose(flipFunc, seeded_rng.randint(len(A_list[0])))(element[0])
                count += 1
                if count > 30:
                    curr = element[0]
            if numParams:
                paramCount, parameterized = scoreFunc(curr)
                score = paramCount
            else:
                score = scoreFunc(curr, beta, A_target)
            currScore = factor * score
            if currScore >= deque.low():
                deque.insert(curr, currScore)
        print(str(i)+"\t"+str(deque.high()) + "\t"+str([x[1] for x in deque]))
    return deque.high(), deque.first()

def generate_key_sequence(network, letters, iterations):
    dictionary = create_dictionary(letters)
    current = seeded_rng.randint(len(network))
    for i in range(iterations):
        g = input(dictionary[str(current)])
        while g != dictionary[str(current)] and g != dictionary[str(current)][::-1]:
            g = input(dictionary[str(current)])
        current = seeded_rng.choice(np.nonzero(network[current])[0])
    print(dictionary)

def compute_triangle_participation(A):
    G = nx.from_numpy_matrix(A)
    cycles_3 = [c for c in nx.cycle_basis(G) if len(c) == 3]
    edges = []
    tri_count = {}
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            if A[i][j] > 0:
                edges.append([i, j])
                tri_count[(i, j)] = 0
    for c in cycles_3:
        for e in edges:
            if set(e).issubset(set(c)):
                tri_count[(e[0], e[1])] += 1
    for e in edges:
        tri_count[(e[0], e[1])] /= min(np.count_nonzero(A[e[0]]) - 1, np.count_nonzero(A[e[1]]) - 1)
    return edges, tri_count, len(cycles_3)

def get_edge_values(A_original, A):
    edge_factors = {}
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            if A_original[i][j] > 0:
                edge_factors[(i, j)] = A[i][j]/A_original[i][j]
    return edge_factors
def create_dictionary(letters):
    result = dict()
    for i in range(len(letters)):
        result[str(i)] = letters[i]
    index = 5
    for i in range(len(letters) - 1):
        for j in range(i + 1, len(letters)):
            result[str(index)] = letters[i] + letters[j]
            index += 1
    return result

def get_lattice_layout():
    graph_pos = {}
    rad_in = .25
    rad_out = 1.2
    left = (-rad_in, 0)
    right = (rad_in, 0)
    top = (0, 2 * rad_in)
    centers = [(np.cos(np.pi / 2 + i  * 2 * np.pi / 5) * rad_out , np.sin(np.pi / 2 + i  * 2 * np.pi / 5) * rad_out) for i in range(5)]
    for i in range(5):
        for j in range(3):
            if j == 0:
                graph_pos[3 *i + j] = (centers[i][0] + top[0], centers[i][1] + top[1])
            if j == 1:
                graph_pos[3 * i + j] = (centers[i][0] + left[0], centers[i][1] + left[1])
            if j == 2:
                graph_pos[3 * i + j] = (centers[i][0] + right[0], centers[i][1] + right[1])
    return graph_pos

def get_modular_layout():
    graph_pos = {}
    top = (0, .8)
    left = (-.5, 0)
    right = (.5, 0)
    rad = .25
    for i in range(5):
        x_off = -np.sin(2 * np.pi / 5 * (i +.5)) * rad
        y_off = -np.cos(2 * np.pi / 5  * (i + .5)) * rad
        graph_pos[i] = (top[0] + x_off, top[1] + y_off)
    for i in range(5):
        x_off = -np.sin(2 * np.pi / 5  * (i + 2)) * rad
        y_off = -np.cos(2 * np.pi / 5  * (i + 2)) * rad
        graph_pos[5 + i] = (right[0] + x_off, right[1] + y_off)
    for i in range(5):
        x_off = np.sin(2 * np.pi / 5  * (i +1.5)) * rad
        y_off = np.cos(2 * np.pi / 5  * (i + 1.5)) * rad
        graph_pos[10 + i] = (left[0] + x_off, left[1] + y_off)
    return graph_pos

def get_optimal_directly(A_target, beta):
    I = np.identity(len(A_target))
    inv_argument = I*(1-np.exp(-beta)) + np.exp(-beta)*A_target
    # print(np.linalg.cond(inv_argument, p='fro'))
    inv = sp.linalg.inv(inv_argument)
    return inv @ A_target

def get_truncated_normal(mean, sd, low, upp):
    return stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

if __name__ == '__main__':

    # weighted = np.loadtxt(semantic+"G_LesMis.csv", delimiter=",")
    # np.fill_diagonal(weighted, 0)
    # weighted = np.maximum(weighted, weighted.T)
    # network0 = deepcopy(weighted)
    # network0[network0 > 0] = 1

    # beta = 5

    # beta_range = np.linspace(1e-3, 2, 500)
    # lambda_cc_range = np.linspace(1e-3, 2, 500)
    # lambda_b_range = np.linspace(1e-3, 2, 500)
    # results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))
    #
    #
    # network0 = get_lattice_graph([3,5])
    # numParams, comps, comps_c, inv_labels, inv_labels_c = get_pickleable_params(network0, include_nonexistent=False)
    # numParams, parameterized = getSymReducedParams(network0, include_nonexistent=False)
    # bounds = [(0, 1) for i in range(numParams)]
    # outcomes = np.zeros((len(beta_range), 1))
    # bounds = [(1e-6, 20) for i in range(2)]
    #
    # for i in range(len(lambda_cc_range)):
    #     print(i)
    #     for j in range(len(lambda_b_range)):
    #         A_init = parameterized([1, lambda_cc_range[i]])
    #         A_learned = learn(A_init, beta_range[j])
    #         score_ext = KL_score_external(A_init, beta_range[j], network0)
    #         score_baseline = KL_score(network0, beta_range[j])
    #         #A_learned = learn(A_init, beta)
    #         #score_ext = KL_score_external(A_init, beta, network0)
    #         #score_baseline = KL_score(network0, beta)
    #         #score_ext = uniformity_cost(network0, A_init, beta)
    #         #score_baseline = uniformity_cost(network0, network0, beta)
    #         results[i][j] = score_ext/score_baseline
    # plt.figure(5)
    # plt.rcParams.update({'font.size': 14})
    # cax = plt.imshow(results, cmap = 'RdBu',extent=[.01, 1, .01, 1], origin='lower', vmax = 1.1, vmin = .9, aspect = 1, norm = mn.MidpointNormalize(midpoint=1))
    # plt.title(r"$\frac{D_{KL}(A || f(A_{in}))}{D_{KL}(A || f(A))}$", size=18)
    # #plt.title(r"$\frac{U(A || f(A^*))}{U(A || f(A))}$", size=18)
    # plt.rcParams.update({'font.size': 14})
    # #plt.xlabel(r"$\lambda_{b}$", size=20)
    # plt.xlabel(r"$\beta$", size=20)
    # plt.ylabel(r"$\lambda _{l}$", size=20)
    # cbar = plt.colorbar(cax, ticks = [.9, .95, 1.0, 1.05, 1.1])
    # cbar.ax.set_yticklabels(['<.9','.95','1.0','1.05', '> 1.1'])
    # # cbar = plt.colorbar(cax, ticks = [.5, .75, 1.0, 1.25, 1.5, 1.75, 2])
    # # cbar.ax.set_yticklabels(['<.5','.75','1.0', '1.25', '1.5', '1.75','>2'])
    # plt.tight_layout()

    # betas = np.linspace(1e-3, 1, 100)
    # scores = np.zeros(len(betas))
    # scores_original = np.zeros(len(betas))
    # #network0 = modular_toys_general(15,3,1,1)
    # network0 = get_lattice_graph([3,5])
    #
    # numParams, comps, comps_c, inv_labels, inv_labels_c = get_pickleable_params(network0, include_nonexistent=False)
    # numParams, parameterized = getSymReducedParams(network0, include_nonexistent=False)
    # bounds = [(0, 1) for i in range(numParams)]
    # outcomes = np.zeros((len(betas), 1))
    # bounds = [(1e-6, 20) for i in range(2)]
    # for i in range(len(betas)):
    #     print(i)
    #     # outcome = op.dual_annealing(KL_modular_toys_general, bounds=bounds,
    #     #                                               args=(15, 3, betas[i]),
    #     #                                               accept = -10, maxiter = 1000, maxfun= 10000)
    #     # A = modular_toys_general(15, 3, outcome.x[0], outcome.x[1])
    #     outcome = op.dual_annealing(pickleable_cost_func, bounds=bounds,
    #                                                             args=(comps, comps_c, inv_labels, inv_labels_c, betas[i], network0, False, True, network0),
    #                                                             accept = -10, maxiter = 1000, maxfun= 100000)
    #     outcomes[i] = (outcome.x/outcome.x[0])[1:]
    #     # outcomes[i] = outcome.x
    #     A = parameterized(outcome.x)
    #     scores_original[i] = KL_score(network0, betas[i])
    #     scores[i] = KL_score_external(A, betas[i], network0)
    #     #scores[i] = uniformity_cost(network0, A, betas[i])
    #     #scores_original[i] = uniformity_cost(network0, network0, betas[i])
    #
    # plt.figure(4, figsize = (5.5,4.5), dpi = 1000)
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel(r'$\beta$')
    # plt.ylabel("Optimal Weight")
    # plt.rcParams.update({'font.size': 16})
    # plt.plot(betas, outcomes[:, 0], label=r'$\lambda_{l}$', color = 'orange')
    # #plt.plot(betas, outcomes[:, 1], label=r'$\lambda_{b}$', color = 'mediumseagreen')
    # plt.legend(frameon = False)
    # plt.tight_layout()
    # plt.savefig('kl_weights_lat.pdf')
    #
    # plt.figure(5, figsize = (5.5,4.5), dpi = 1000)
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel(r'$\beta$')
    # #plt.ylabel('Uniformity Cost, ' + r'$U(A||f(A^{*}))$')
    # plt.ylabel('KL Divergence, ' + r'$D_{KL}(A||f(A_{in}))$')
    # #plt.ylabel('KL Divergence')
    # plt.rcParams.update({'font.size': 16})
    # plt.plot(betas, scores_original, label='Original ('+r'$A_{in} = A$'+')', color = 'lightgrey')
    # plt.plot(betas, scores, label='Optimized (' + r'$A_{in} = A^{*}$' + ')', color = 'black')
    # plt.legend(frameon = False)
    # plt.tight_layout()
    # plt.savefig('kl_scores_lat.pdf')
    #
    # print('good beta', betas[np.argmax(scores)])

    beta = .05
    network0 = modular_toys_general(15,3,1,1)
    #network0 = get_lattice_graph([3,5])
    # numParams, comps, comps_c, inv_labels, inv_labels_c = get_pickleable_params(network0, include_nonexistent= False)
    # # print(comps)
    # # print(comps_c)
    # numParams, parameterized = getSymReducedParams(network0, include_nonexistent= False)
    # bounds = [(0, 1) for j in range(numParams)]
    # outcome = op.dual_annealing(pickleable_cost_func, bounds = bounds,
    #                             args=(comps, comps_c, inv_labels, inv_labels_c, betas[i], network0, False, True, network0),
    #                             accept = -50, maxiter = 1500, maxfun= 1000000)
    #
    # A = parameterized(outcome.x)
    # # A /= np.sum(A)
    # # A *= 60
    # A = normalize(A)
    # # #print(outcome.x/ outcome.x[0])
    # # print(KL_score_external(A, beta, network0), KL_score(network0, beta))
    # #network0 = normalize(network0)
    # # network0 /= np.sum(network0)
    # # network0 *= 60

    # biased = modular_toys_general(15,3, .5 ,1.5)
    # plt.figure(0)
    # cmap = plt.get_cmap("RdGy")
    # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # G_0 = nx.from_numpy_matrix(network0)
    #
    #
    # graph_pos = nx.spring_layout(G_0, iterations = 1000, k = .5)
    # graph_pos = get_modular_layout()
    #
    # edgewidth = [max(2, 4 * abs(d['weight'])**1)  for (u, v, d) in G_0.edges(data=True)]
    # #edgecolor = [(0, 0, 0, min(.8 * d['weight'], 1)) for (u, v, d) in G_0.edges(data=True)]
    # G_temp = nx.from_numpy_matrix(biased)
    # all_cols = mcolors.CSS4_COLORS
    # edgecolor = []
    # for (u, v, d) in G_temp.edges(data=True):
    #     if d['weight'] == 0.5:
    #         edgecolor.append(all_cols['orange'])
    #         #edgecolor.append((0, 0.6, 0, 1))
    #     elif d['weight']== 1.5:
    #         edgecolor.append(all_cols['mediumseagreen'])
    #         #edgecolor.append((1, 0.7, .4, 1))
    #     else:
    #         edgecolor.append(all_cols['grey'])
    #         #edgecolor.append((0, 0, 0, 1))
    #
    # nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal), with_labels= False, node_color = 'lightblue')
    # nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
    # ax = plt.gca()
    # ax.collections[0].set_edgecolor("#000000")
    #
    # plt.axis('off')
    # plt.colorbar(sm, ticks=np.linspace(-1, 1, 6))


    # plt.figure(1)
    # learned_0 = learn(network0, beta)
    # learned_0 = unnormalize(learned_0)
    # learned_0 /= np.sum(learned_0)
    # learned_0 *= 60
    #
    # print("UNIFORMITY ORIGINAL", uniformity_cost(network0, network0, beta))
    # G_0 = nx.from_numpy_matrix(learned_0)
    # edgewidth = [max(2, 4 * d['weight']**1)  for (u, v, d) in G_0.edges(data=True)]
    # edgecolor = [(0,0, 0, min(.8  * d['weight'], 1))  for (u, v, d) in G_0.edges(data=True)]
    # nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal), with_labels= False, node_color = 'lightblue')
    # nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
    # ax = plt.gca()
    # ax.collections[0].set_edgecolor("#000000")
    #
    # plt.axis('off')
    #
    #
    # plt.figure(2)
    # G_0 = nx.from_numpy_matrix(A)
    # edgewidth = [max(2, 4 * d['weight']**1)  for (u, v, d) in G_0.edges(data=True)]
    # edgecolor = [(0, 0, 0, min(.8  * d['weight'], 1))for (u, v, d) in G_0.edges(data=True)]
    # nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal), with_labels= False, node_color = 'lightblue')
    # nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
    # ax = plt.gca()
    # ax.collections[0].set_edgecolor("#000000")
    #
    # plt.axis('off')
    # #plt.colorbar(sm, ticks=np.linspace(0, 1, 6))
    #
    # plt.figure(3)
    # learned_A = learn(A, beta)
    # learned_A = unnormalize(learned_A)
    # #learned_A[network0 == 0] = 0
    # learned_A /= np.sum(learned_A)
    # learned_A *= 60
    #
    # print(check_symmetric(learned_A))
    # print("UNIFORMITY OPTIMIZED", uniformity_cost(network0, A, beta))
    # G_0 = nx.from_numpy_matrix(learned_A)
    # edgewidth = [max(2, 4 * d['weight']**1)  for (u, v, d) in G_0.edges(data=True)]
    # edgecolor = [(0, 0, 0, min(.8 * d['weight'], 1)) for (u, v, d) in G_0.edges(data=True)]
    # nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal), with_labels= False, node_color = 'lightblue')
    # nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
    # ax = plt.gca()
    # ax.collections[0].set_edgecolor("#000000")
    #
    # plt.axis('off')
    #
    plt.show()