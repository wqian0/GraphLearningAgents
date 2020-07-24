import copy
from colour import Color
#import numpy as np
import mynumpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import colors
from operator import itemgetter
from graphviz import Digraph
import os
import heapq
import tempfile
import itertools
# import dill as pk
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
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

rng = np.random.RandomState()
#seeded_rng = np.random.RandomState(17310145)
seeded_rng = rng
N = 12
N_internal = 24
sources = 10

languages = "C:/Users/billy/PycharmProjects/GraphLearningAgents/graphs_Language_share/"
music = "C:/Users/billy/PycharmProjects/GraphLearningAgents/graphs_Music_share/"
web = "C:/Users/billy/PycharmProjects/GraphLearningAgents/graphs_Web_share/"
social = "C:/Users/billy/PycharmProjects/GraphLearningAgents/graphs_Social_share/"
citation = "C:/Users/billy/PycharmProjects/GraphLearningAgents/graphs_Citation_share/"
semantic = "C:/Users/billy/PycharmProjects/GraphLearningAgents/graphs_Semantic_share/"

def matrix(m, n, val):
    M = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(val)
        M.append(row)
    return np.array(M)

def get_random_modular(n, modules, directedEdges, p, getCommInfo=False):
    pairings = {}
    assignments = np.zeros(n, dtype = int)
    for i in range(modules):
        pairings[i] = []
    adjMatrix = matrix(n, n, 0)
    for i in range(n):
        randomModule = seeded_rng.randint(0, modules)
        pairings[randomModule].append(i)
        assignments[i] = randomModule

    def add_modular_edge(module = -1):
        if module == -1:
            randomComm = seeded_rng.randint(0, modules)
        else:
            randomComm = module
        while len(pairings[randomComm]) < 2:
            randomComm = seeded_rng.randint(0, modules)
        selection = seeded_rng.choice(pairings[randomComm], 2, replace=True)
        while adjMatrix[selection[0]][selection[1]] != 0:
            randomComm = seeded_rng.randint(0, modules)
            while len(pairings[randomComm]) < 2:
                randomComm = seeded_rng.randint(0, modules)
            selection = seeded_rng.choice(pairings[randomComm], 2, replace=True)
        adjMatrix[selection[0]][selection[1]] += 1

    def add_random_edge(): #adds edge anywhere
        randEdge = seeded_rng.choice(n, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0:
            randEdge = seeded_rng.choice(n, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1
    inModuleEdges = round(directedEdges * p)
    randEdges = directedEdges - inModuleEdges
    for i in range(inModuleEdges):
        add_modular_edge()
    for i in range(randEdges):
        add_random_edge()
    if getCommInfo:
        return adjMatrix, pairings, assignments
    else:
        return adjMatrix

def learn(A, beta):
    A = normalize(A)
    inverse_argument = np.identity(len(A)) - np.exp(-beta)*A
    inverse = sp.linalg.inv(inverse_argument)
    return normalize((1-np.exp(-beta))*(A @ inverse))

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
    B = deepcopy(A)
    for i in range(len(A)):
        #B[i] /= ((B[i])[np.nonzero(B[i])]).mean()
        B[i] /= np.amax(B[i])
    return B/np.amax(B)
#Assumes undirected input!

def getNumEdges(A):
    return np.count_nonzero(A) / 2
def get_stationary3(A):
    output = np.count_nonzero(A, axis = 0) / (2 * getNumEdges(A))
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
def KL_Divergence(U, V):
    U = normalize(U)
    V = normalize(V)
    pi = get_stationary2(U)
    combined = np.einsum('i, ij -> ij', pi, U)
    logged = np.log(V/U)
    logged[U == 0] = 0
    result = combined.T @ logged
    return -np.trace(result)

def create_agent_network(N, numSources, edges):
    adjMatrix = np.zeros((N, N), dtype = float)
    for i in range(sources, N):
        randIndex = seeded_rng.randint(N)
        while adjMatrix[randIndex][i] != 0 or i == randIndex:
            randIndex = seeded_rng.randint(N)
        adjMatrix[randIndex][i] += 1.0
    for i in range(edges - N + sources):
        randEdge = seeded_rng.choice(N, 2, replace=False)
        while adjMatrix[randEdge[0]][randEdge[1]] != 0 or randEdge[1] < numSources:
            randEdge = seeded_rng.choice(N, 2, replace=False)
        adjMatrix[randEdge[0]][randEdge[1]] += 1.0
    return adjMatrix

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
    #perm = seeded_rng.permutation(np.arange(N))
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

def modular_toy_paper():
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

def agent_network_sim(network, agent_networks_init, iterations, betas):
    divergences = np.zeros((N, iterations))
    agent_networks = copy.deepcopy(agent_networks_init)
    agent_networks_MLE = copy.deepcopy(agent_networks_init)
    for i in range(iterations):
        print("iteration \t" +str(i))
        for j in range(N):
            external_input = 0
            for k in np.nonzero(network[:, j])[0]:
                external_input += network[k][j] * normalize(agent_networks[k])
            if len(np.nonzero(network[:, j])[0]) != 0:
                agent_networks_MLE[j] += external_input
                agent_networks[j] = normalize(learn(normalize(agent_networks_MLE[j]), betas[j]))
            divergences[j][i] = KL_Divergence(agent_networks_init[0], normalize(agent_networks[j]))
    return agent_networks, divergences

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol) and np.allclose(np.diag(a), np.zeros(len(a)))

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

def printMatrixToFile(M, file):
    for r in range(len(M)):
        for c in range(len(M[0])):
            file.write(str(M[r][c]) + "\t")
        file.write("\n")
    file.write("\n")

def KL_score(A, beta, A_target = None):
    return KL_Divergence(A, learn(A, beta))

def KL_score_ext_zipped(input, beta):
    cc_bias, b_bias = input
    if cc_bias < 1e-8 or b_bias < 1e-8:
        return 100
    return KL_score_external(biased_modular(cc_bias, b_bias), beta, modular_toy_paper())

counterOpt = 0
mod_network = modular_toy_paper()
def KL_score_ext_full(input, beta):
    global counterOpt, mod_network
    counterOpt += 1
    if counterOpt % 1000 == 0:
        print(counterOpt)
    iu = np.triu_indices(15, k = 1)
    A = np.zeros((15,15))
    A[iu] = input
    A = np.maximum(A, A.T)
    return KL_score_external(A, beta, mod_network)

def KL_score_external(A_input, beta, A_target):
    return KL_Divergence(A_target, learn(A_input, beta))

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

def get_structurally_symmetric(A):
    autos = get_automorphisms(A)
    disj_set = uf.UnionFind()
    for i in range(len(A)):
        disj_set.add(i)
    for i in range(len(A)):
        print(i)
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

#only works perfectly for symmetric
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


def get_KL_ext_general(A_target):
    numParams, parameterized = getSymReducedParams(A_target)
    def cost_func(input, beta):
        return KL_score_external(parameterized(input), beta, A_target)
    return numParams, cost_func, parameterized

def get_pickleable_params(A, include_nonexistent = True):
    comps, comp_maps, edge_labels, inv_labels = unique_edges(A, 0)
    if include_nonexistent:
        A_c = get_complement_graph(A)
        comps_c, comp_maps_c, edge_labels_c, inv_labels_c = unique_edges(A_c, 0)
        return len(comps) + len(comps_c), comps, comps_c, inv_labels, inv_labels_c
    return len(comps), comps, None, inv_labels, None
def pickleable_cost_func(input, comps, comps_c, inv_labels, inv_labels_c, beta, A_target, include_nonexistent):
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
    return KL_score_external(B, beta, A_target)

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

def grad_zipped(input, P_0, pi, eta, J, I):
    iu = np.triu_indices(15, k=1)
    A = np.zeros((15, 15))
    A[iu] = input
    A = np.maximum(A, A.T)
    output = gradient(P_0, pi, A, eta, J, I)
    return output[iu]


def cost_func_zipped(input, P_0, pi, eta, J, I):
    iu = np.triu_indices(N_internal, k=1)
    A = np.zeros((N_internal, N_internal))
    A[iu] = input
    A = np.maximum(A, A.T)
    cost = cost_func(P_0, pi, A, eta, J, I)
    return cost

def Q_inv(A, beta, depth):
    nu = np.exp(-beta)
    result = 0
    for i in range(depth):
        result += np.linalg.matrix_power(nu * A, i)
    return result

def cost_func(P_0, pi, A, eta, J, I):
    AJ = A @ J
    P_f = A / AJ
    P_f[np.isnan(P_f)] = 0
    Q = sp.linalg.inv(I - eta * P_f)
    prod = P_f @ Q
    M2 = np.log(prod/ P_0)
    M2[P_0 == 0] = 0
    combined = np.einsum('i, ij -> ij', pi, P_0)
    return -np.trace(combined.T @ M2)

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

def cost_X(P_0, pi, X, eta, J, I):
    A = sigmoid(X)
    AJ = A @ J
    P_f = A / AJ
    Q = sp.linalg.inv(I - eta * P_f)
    M2 = np.log(P_f @ Q / P_0)
    M2[np.isnan(M2)] = 0
    M2[M2 == np.inf] = 0
    if(np.any(M2 == -np.inf)):
        print("WHAT")
        return 10000
    combined = np.einsum('i, ij -> ij', pi, P_0)
    return -np.trace(combined.T @ M2)

def cost_X_zipped(input, P_0, pi, eta, J, I):
    iu = np.triu_indices(15, k=1)
    X = np.zeros((15, 15))
    X[iu] = input
    X = np.maximum(X, X.T)
    cost = cost_X(P_0, pi, X, eta, J, I)
    return -np.log(1-eta) + cost
def sigmoid(X):
    return 1/(1+np.exp(-X))
def inv_sigmoid(Y):
    return np.log(Y/(1-Y))

def grad_X(P_0, pi, X, eta, J, I):
    A = sigmoid(X)
    AJ = A @ J
    P_f = A / AJ
    Q = sp.linalg.inv(I - eta * P_f)
    combined = np.einsum('i, ij -> ij', pi, P_0)
    R = combined / (P_f @ Q)
    S = (eta * Q.T @ P_f.T - I) @ (R @ Q.T) / (AJ * AJ)
    return A * (1-A) * (S * AJ - ((S * A) @ J))
    return output

def grad_X_zipped(input, P_0, pi, eta, J, I):
    iu = np.triu_indices(15, k=1)
    X = np.zeros((15, 15))
    X[iu] = input
    X = np.maximum(X, X.T)
    output = grad_X(P_0, pi, X, eta, J, I)
    return output[iu]

def descend_X(P_0, pi, beta, rate, iterations):
    eta = np.exp(-beta)
    I = np.identity(len(P_0))
    J = np.ones((len(P_0), len(P_0)))
    count = 0
    norm = np.inf
    mod = normalize(modular_toy_paper())
    X = inv_sigmoid(mod)
    while count < iterations or norm > 1e-8:
        descent = rate * grad_X(P_0, pi, X, eta, J, I)
        X -= descent
        count += 1
        norm = sp.linalg.norm(descent)
        if count % 1000 == 0:
            print(str(norm) + "\t" + str(count) + "\t" + str(
                KL_score_external(np.exp(X), beta, mod)))
    A = sigmoid(X)
    result = A / (A @ J)
    return result
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
            print(str(norm)+"\t"+str(count)+"\t"+str(np.min(A))+"\t"+str(KL_score_external(A, .3, modular_toy_paper()))+"\t"+str(cost))
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
    print(str(bestVal)+"\t BEST VAL!")
    return bestVal,  best

def uniformity_cost(P_0, A, beta):
    learned = learn(A, beta)
    terms = learned[P_0 > 0].flatten()
    diffs = np.subtract.outer(terms, terms)
    return np.sqrt(np.sum(diffs * diffs))

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

def get_truncated_normal(mean, sd, low, upp):
    return stats.truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
if __name__ == '__main__':
    # A_target = modular_toy_paper()
    #
    # beta_range = np.linspace(1e-3, 2, 500)
    # lambda_cc_range = np.linspace(1e-3, 2, 500)
    # lambda_b_range = np.linspace(1e-3, 2, 500)
    # results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))
    # beta = .5
    # for i in range(len(lambda_cc_range)):
    #     print(i)
    #     for j in range(len(lambda_b_range)):
    #         A_init = biased_modular(lambda_cc_range[i], lambda_b_range[j])
    #         # score_ext = KL_score_external(A_init, beta_range[i], A_target)
    #         # score_baseline = KL_score(A_target, beta_range[i])
    #         # results[i][j] = score_ext/score_baseline
    #         results[i][j] = frobenius_norm_cost(A_target, A_init, beta) / frobenius_norm_cost(A_target, A_target, beta)
    #
    # pk.dump([lambda_cc_range, lambda_b_range, results], open("frobenius Cost lambda-lambda heatmap 0.5.pickle", "wb"))
    # plt.figure(5)
    # cax = plt.imshow(results, cmap='RdBu', extent=[.01, 2, .01, 2], origin='lower', norm = mn.MidpointNormalize(midpoint=1), vmin = 0.9, vmax = 1.001, aspect=1)
    # plt.title("Frobenius Cost", size=16)
    # plt.ylabel(r"$\lambda_{cc}$", size=16)
    # plt.xlabel(r"$\lambda _{b}$", size=16)
    # plt.colorbar(cax)
    # cbar = plt.colorbar(cax, ticks=[.2, .4, .6, .8, 1])
    # cbar.ax.set_yticklabels(['.2', '.4', '.6', '.8', '>1'])

    #network0 = np.loadtxt(social+"G_karate.csv", delimiter=',')

    A_list = []
    for i in [2,3,4,6]:
        network0 = modular_toys_general(N_internal, i, 1, 1)
        beta = .05
        eta = np.exp(-beta)
        J = np.ones((len(network0), len(network0)))
        I = np.identity(len(network0))
        pi = get_stationary3(network0)
        bounds = [(0, 10) for i in range(2)]
        #numParams, comps, comps_c, inv_labels, inv_labels_c = get_pickleable_params(network0, include_nonexistent= True)
        #print(comps)
        #numParams, parameterized = getSymReducedParams(network0, include_nonexistent= True)
        # outcome = op.differential_evolution(pickleable_cost_func, bounds=bounds, tol=1e-10, maxiter = 100000, workers=-1,
        #                                    args=(comps, comps_c, inv_labels, inv_labels_c, beta, network0, True), disp = True)
        # A = parameterized(list(outcome.x))
        # outcome = op.differential_evolution(cost_func_zipped, bounds = bounds, args = (normalize(network0), pi, eta, J, I), workers = -1, disp = True, maxiter = 100000, tol = 1e-10)
        outcome = op.differential_evolution(KL_modular_toys_general, bounds= bounds, args = (N_internal, i, beta), workers = -1, disp = True, maxiter = 100000, tol = 1e-10)
        A = modular_toys_general(N_internal, i, outcome.x[0], outcome.x[1])
        print(outcome.x)

        print("YEET")
        print(KL_score_external(A, beta, network0), KL_score(network0, beta))

        network0 = normalize(network0)
        plt.figure(0)
        cmap = plt.get_cmap("binary")
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        G_0 = nx.from_numpy_matrix(network0)
        graph_pos = nx.spring_layout(G_0, iterations = 100)
        edgewidth = [max(.25 ,4 * (d['weight'])) for (u, v, d) in G_0.edges(data=True)]
        edgecolor = [cmap(max(.1, 4 * d['weight'])) for (u, v, d) in G_0.edges(data=True)]
        nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal))
        nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
        #plt.colorbar(sm, ticks=np.linspace(0, 1, 6))


        plt.figure(1)
        learned_0 = learn(network0, beta)
        G_0 = nx.from_numpy_matrix(learned_0)
        edgewidth = [max(.25 ,4 * (d['weight'])) for (u, v, d) in G_0.edges(data=True)]
        edgecolor = [cmap(max(.1, 4 * d['weight'])) for (u, v, d) in G_0.edges(data=True)]
        nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal))
        nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
        #plt.colorbar(sm, ticks=np.linspace(0, 1, 6))


        plt.figure(2)
        G_0 = nx.from_numpy_matrix(A)
        edgewidth = [max(.25 ,4 * (d['weight']) ) for (u, v, d) in G_0.edges(data=True)]
        edgecolor =[cmap(max(.1, 4 * d['weight'])) for (u, v, d) in G_0.edges(data=True)]
        nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal))
        nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
        #plt.colorbar(sm, ticks=np.linspace(0, 1, 6))

        plt.figure(3)
        learned_A = learn(A, beta)
        G_0 = nx.from_numpy_matrix(learned_A)
        edgewidth = [max(.25 ,4 * (d['weight']) ) for (u, v, d) in G_0.edges(data=True)]
        edgecolor = [cmap(max(.1, 4 * d['weight'])) for (u, v, d) in G_0.edges(data=True)]
        nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal))
        nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, connectionstyle='arc3, rad = 0.1', width=edgewidth)
        #plt.colorbar(sm, ticks=np.linspace(0, 1, 6))

        # print(pd.DataFrame(network0))
        # print(pd.DataFrame(learned_0))
        # print(pd.DataFrame(A))
        # print(pd.DataFrame(learned_A))
        A_list.append(A)
        plt.show()

    pk.dump(A_list, open("Karate .05.pickle","wb"))
