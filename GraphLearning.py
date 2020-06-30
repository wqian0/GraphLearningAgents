import copy
from colour import Color
#import numpy as np
import mynumpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from graphviz import Digraph
import os
import tempfile
import itertools
import pickle as pk
from copy import deepcopy
import copy
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

import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState()
#seeded_rng = np.random.RandomState(17310145)
seeded_rng = rng
N = 15
N_internal = 15
sources = 10

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
    inverse = np.linalg.pinv(inverse_argument)
    return normalize((1-np.exp(-beta))*np.dot(A, inverse))

def get_stationary(A):
    lam, vec = sp.linalg.eig(A, left=True, right=False)
    idx = np.argmin(np.abs(lam - 1))
    w = np.real(vec[:, idx])
    return w / w.sum().real

def get_stationary2(A):
    P10000 = np.linalg.matrix_power(A, 10000)
    P10001 = np.dot(P10000, A)
    while not np.allclose(P10000, P10001):
        P10001 = np.dot(P10001, A)
        #print("STUCK REAL")
    return P10001[0]
def normalize(A, delLoops = False):
    B = deepcopy(A)
    J = np.ones((len(A), len(A)))
    output = B / (B @ J)
    output[np.isnan(output)] = 0
    return output

def KL_Divergence(U, V):
    U = normalize(U)
    V = normalize(V)
    pi = get_stationary2(U)
    result = 0
    for i in range(len(U)):
        for j in range(len(U)):
            if not np.isclose(U[i][j], 0, rtol = 1e-16) and not np.isclose(V[i][j], 0, rtol = 1e-16):
                result += pi[i] * U[i][j] * np.log(V[i][j]/U[i][j])
    return -result

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

def get_automorphisms(A):
    IG = ig.Graph.Adjacency(A.tolist())
    return np.transpose(np.array(IG.get_automorphisms_vf2()))

def get_regular_graph(N, d):
    return nx.to_numpy_matrix(nx.random_regular_graph(d, N))
def get_structurally_symmetric(A):
    autos = get_automorphisms(A)
    # print(autos)
    disj_set = uf.UnionFind()
    for i in range(len(A)):
        disj_set.add(i)
    for i in range(len(A)):
        # print(i)
        for j in range(len(autos[0])):
            disj_set.union(i, autos[i][j])
    return disj_set.components(), disj_set.component_mapping()

#input: symmetric adj matrix
def compute_line_graph(A):
    G_input = nx.from_numpy_matrix(A)
    print("AHH")
    return nx.to_numpy_matrix(nx.line_graph(G_input))

def unique_edges(A, beta, A_target = None):
    edge_labels, inv_labels, line_graph = compute_line_graph_details(A, 30)
    components, comp_mappings = get_structurally_symmetric(line_graph)
    return components, comp_mappings, edge_labels, inv_labels


def grad_zipped(input, P_0, eta, J, I):
    iu = np.triu_indices(15, k=1)
    A = np.zeros((15, 15))
    A[iu] = input
    A = np.maximum(A, A.T)
    output = gradient(P_0, A, eta, J, I)
    return output[iu]


def cost_func_zipped(input, P_0, eta, J, I):
    iu = np.triu_indices(15, k=1)
    A = np.zeros((15, 15))
    A[iu] = input
    A = np.maximum(A, A.T)
    cost = cost_func(P_0, A, eta, J, I)
    return cost

def cost_func(P_0, A, eta, J, I):
    AJ = A @ J
    P_f = A / AJ
    P_f[np.isnan(P_f)] = 0
    Q = np.linalg.pinv(I - eta * P_f)
    prod = P_f @ Q
    M2 = np.log(prod/ P_0)
    M2[P_0 == 0] = 0
    if(np.any(M2 == -np.inf)):
        print("AGH")
        return 1000
    return -np.trace(P_0.T @ M2)

def gradient(P_0, A, eta, J, I):
    AJ = A @ J
    P_f = A / AJ
    P_f[np.isnan(P_f)] = 0
    Q = np.linalg.pinv(I- eta * P_f)
    R = P_0 / (P_f @ Q)
    R[np.isnan(R)] = 0
    S = (eta * Q.T @ P_f.T - I) @ (R @ Q.T)/ (AJ * AJ)
    S[np.isnan(S)] = 0
    output = S * AJ - (S*A) @ J
    return output

def cost_X(P_0, X, eta, J, I):
    A = sigmoid(X)
    AJ = A @ J
    P_f = A / AJ
    Q = np.linalg.inv(I - eta * P_f)
    M2 = np.log(P_f @ Q / P_0)
    M2[np.isnan(M2)] = 0
    M2[M2 == np.inf] = 0
    if(np.any(M2 == -np.inf)):
        print("WHAT")
        return 10000
    return -np.trace(P_0.T @ M2)

countCost = 0
def cost_X_zipped(input, P_0, eta, J, I):
    global countCost
    iu = np.triu_indices(15, k=1)
    X = np.zeros((15, 15))
    X[iu] = input
    X = np.maximum(X, X.T)
    cost = cost_X(P_0, X, eta, J, I)
    countCost += 1
    if countCost % 10000 == 0:
        print(cost)
    return cost
def sigmoid(X):
    return 1/(1+np.exp(-X))
def inv_sigmoid(Y):
    return np.log(Y/(1-Y))

def grad_X(P_0, X, eta, J, I):
    A = sigmoid(X)
    AJ = A @ J
    P_f = A / AJ
    Q = np.linalg.inv(I - eta * P_f)
    R = P_0 / (P_f @ Q)
    S = (eta * Q.T @ P_f.T - I) @ (R @ Q.T) / (AJ * AJ)
    return A *(1-A) * (S * AJ - ((S * A) @ J))
    return output
def grad_X_zipped(input, P_0, eta, J, I):
    iu = np.triu_indices(15, k=1)
    X = np.zeros((15, 15))
    X[iu] = input
    X = np.maximum(X, X.T)
    output = grad_X(P_0, X, eta, J, I)
    return output[iu]

def descend_X(P_0, beta, rate, iterations):
    eta = np.exp(-beta)
    I = np.identity(len(P_0))
    J = np.ones((len(P_0), len(P_0)))
    count = 0
    norm = np.inf
    mod = normalize(modular_toy_paper())
    X = np.ones((len(P_0), len(P_0)))
    while count < iterations or norm > 1e-8:
        descent = rate / (len(P_0)) * grad_X(P_0, X, eta, J, I)
        X -= descent
        count += 1
        norm = np.linalg.norm(descent)
        if count % 1000 == 0:
            print(str(norm) + "\t" + str(count) + "\t" + str(
                KL_score_external(np.exp(X), .3, mod)))
    A = sigmoid(X)
    result = A / (A @ J)
    return result
def descend(P_0, beta, rate, iterations):
    eta = np.exp(-beta)
    I = np.identity(len(P_0))
    A = biased_modular(.25, .8)
    J = np.ones((len(P_0), len(P_0)))
    count = 0
    norm = np.inf
    mod = modular_toy_paper()
    while count < iterations or norm > 1e-8:
        descent = rate / (len(P_0)** 2) * gradient(P_0, A, eta, J, I)
        A -= descent
        count += 1
        norm = np.linalg.norm(descent)
        A[A < 0] = 0
        if count % 1000 == 0:
            cost = cost_func(P_0, A, eta, J, I)
            print(str(norm)+"\t"+str(count)+"\t"+str(np.min(A))+"\t"+str(KL_score_external(A, .3, modular_toy_paper()))+"\t"+str(cost))
    result = A / np.dot(A, J)
    return result
def compute_line_graph_details(A, edges):
    edge_labels = dict()
    count = 0
    for r in range(len(A)-1):
        for c in range(r+1, len(A)):
            if A[r][c] == 1.0:
                edge_labels[(r,c)] = count
                count += 1
    inv_labels = {v: k for k, v in edge_labels.items()}
    keys = list(edge_labels.keys())
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
    G_input = nx.from_numpy_matrix(A)
    return nx.to_numpy_matrix(nx.complement(G_input))
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
def optimize(A, beta, iterations, scoreFunc, flipFunc, minimize = True, A_target = None, line_graph = False):
    bestVal = float('inf')
    collecting = False
    curr = deepcopy(A)
    best = deepcopy(A)
    factor = 1
    if not minimize:
        factor = -1
    for i in range(iterations):
        if line_graph:
            components, comp_mappings, edge_labels, inv_labels = scoreFunc(curr, beta, A_target)
            score = factor * len(components)
        else:
            score = scoreFunc(curr, beta, A_target)
        currScore = factor * score
        if currScore < bestVal and isConnected(curr):
            bestVal = currScore
            if line_graph:
                best = deepcopy(curr), components, comp_mappings, edge_labels, inv_labels
            else:
                best = deepcopy(curr)
        if line_graph:
            curr = compose(flipFunc, seeded_rng.randint(2 * len(A)))(best[0])
        else:
            curr = compose(flipFunc, seeded_rng.randint(2 * len(A)))(best)
        count = 0
        while not isConnected(curr):
            if line_graph:
                curr = compose(flipFunc, seeded_rng.randint(2 * len(A)))(best[0])
            else:
                curr = compose(flipFunc, seeded_rng.randint(2 * len(A)))(best)
            count += 1
            if count > 30:
                return bestVal, best
        print(str(i)+"\t"+str(currScore)+"\t"+str(bestVal))
    print(str(bestVal)+"\t BEST VAL!")
    return bestVal,  best

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
    P_0 = normalize(modular_toy_paper())
    eta = np.exp(-.05)
    iu = np.triu_indices(15, k = 1)
    J = np.ones((15,15))
    I = np.identity(15)
    print(cost_func(P_0, P_0, eta, J, I))
    print(cost_func_zipped(P_0[iu], P_0, eta, J ,I))
    print(cost_func(P_0, biased_modular(.25,.8), eta, J, I))
    bounds = [(0, 1) for i in range(105)]
    outcome = op.differential_evolution(cost_func_zipped, bounds = bounds, args = (P_0, eta, J, I), tol = 1e-6 , workers = -1, disp = True, maxiter = 25000)
    #outcome = op.basinhopping(cost_func_zipped, np.ones(105), disp = True, minimizer_kwargs={"jac":grad_zipped, "method": "L-BFGS-B", "args" : (P_0, eta, J, I), "bounds":bounds})
    A = np.zeros((15,15))
    A[iu] = outcome.x
    A = np.maximum(A, A.T)
    score = KL_score_external(A, .05, P_0)
    scores_original = KL_score(modular_toy_paper(), .05)
    print(str(.3)+"\t"+str(normalize(A)) +"\t"+str(score))
    P_f = normalize(A)
    print(KL_score_external(P_f, .05, modular_toy_paper()))
    plt.figure(1)
    cmap = plt.get_cmap("hot_r")
    norm = mpl.colors.Normalize(vmin=0, vmax=.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    G_0 = nx.from_numpy_matrix(P_0)
    graph_pos = nx.spring_layout(G_0, iterations=50)
    edgewidth = [(4*d['weight']) for (u, v, d) in G_0.edges(data=True)]
    edgecolor =[cmap(4 * d['weight']) for (u, v, d) in G_0.edges(data=True)]
    nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal))
    nx.draw_networkx_edges(G_0, graph_pos,  edge_color= edgecolor, width = edgewidth)
    plt.colorbar(sm, ticks=np.linspace(0,.25, 6))

    plt.figure(2)
    G_f = nx.from_numpy_matrix(P_f)
    edgewidth = [(4*d['weight']) for (u, v, d) in G_f.edges(data=True)]
    edgecolor =[cmap(4 * d['weight']) for (u, v, d) in G_f.edges(data=True)]
    nx.draw_networkx(G_f, graph_pos, width=np.zeros(N_internal))
    nx.draw_networkx_edges(G_f, graph_pos,  edge_color= edgecolor, width = edgewidth)
    plt.colorbar(sm, ticks=np.linspace(0, .25, 6))

    pk.dump(P_f, open(r"optimal_modular_25k_point05.pickle", "wb"))
    #
    # print(P_f[1][0])
    # print(P_f[1][2])
    # print(P_f[3][4])
    # betas = np.linspace(1e-8, 3, 100)
    # params = np.zeros((len(betas), 2))
    # scores = np.zeros(len(betas))
    # scores_original = np.zeros(len(betas))
    #
    # for i in range(len(betas)):
    #     outcome = op.minimize(KL_score_ext_zipped, [1, 1], method='Nelder-Mead', args=(betas[i],), tol=1e-10,
    #                           options={"maxiter": 1000})
    #     params[i] = np.array(outcome.x)
    #     scores[i] = KL_score_ext_zipped(outcome.x, betas[i])
    #     scores_original[i] = KL_score(modular_toy_paper(), betas[i])
    #     print(str(betas[i]) + "\t" + str(outcome.x) + "\t" + str(scores[i]))
    #
    # print(KL_score_external(modular_toy_paper(), .3, modular_toy_paper()))
    #
    # plt.figure(3)
    # plt.scatter(betas, params[:, 0], label="cross-cluster bias", s=15)
    # plt.scatter(betas, params[:, 1], label="boundary bias", s=15)
    # plt.legend()
    #
    # plt.figure(4)
    # plt.plot(betas, scores, label="optimized")
    # plt.plot(betas, scores_original, label="original")
    # plt.legend()
    # plt.figure(5)
    # plt.plot(betas, scores / scores_original)
    #
    # cmap = plt.get_cmap("hot_r")
    # norm = mpl.colors.Normalize(vmin=0,vmax=.25)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.figure(6)
    # G_f = nx.from_numpy_matrix(normalize(biased_modular(params[9][0], params[9][1])))
    # edgewidth = [(5 * d['weight'])  for (u, v, d) in G_f.edges(data=True)]
    # graph_pos = nx.spring_layout(G_f, iterations = 50)
    # nx.draw_networkx(G_f, graph_pos, width=np.zeros(N_internal))
    # nx.draw_networkx_edges(G_f, graph_pos, width=edgewidth)
    # plt.colorbar(sm, ticks=np.linspace(0, .25, 6))
    #
    # print(pd.DataFrame(normalize(biased_modular(params[9][0], params[9][1]))))
    #
    # pk.dump([betas, params, scores, scores_original], open(r"Modular Symmetry-Reduced Optimized Per Beta.pickle","wb"))
    # # print(KL_score_external(biased_modular(.25,.8), .3, modular_toy_paper()))
    # P_0 = normalize(modular_toy_paper())
    # # P_f = pk.load(open(r"optimal_modular_point3.pickle", "rb"))
    # P_f = descend_X(P_0, .3, 1e-1, 100000)
    #
    # plt.figure(0)
    # cmap = plt.get_cmap("hot_r")
    # norm = mpl.colors.Normalize(vmin=0,vmax=.25)
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # G_0 = nx.from_numpy_matrix(P_0)
    # graph_pos = nx.spring_layout(G_0, iterations=50)
    # edgewidth = [4 * d['weight'] for (u, v, d) in G_0.edges(data=True)]
    # edgecolor =[cmap(4 * d['weight']) for (u, v, d) in G_0.edges(data=True)]
    # nx.draw_networkx(G_0, graph_pos, width=np.zeros(N_internal))
    # nx.draw_networkx_edges(G_0, graph_pos,  edge_color= edgecolor)
    # plt.colorbar(sm, ticks=np.linspace(0,.25, 6))
    #
    # plt.figure(1)
    # G_f = nx.from_numpy_matrix(P_f)
    # graph_pos = nx.spring_layout(G_0, iterations=50)
    # edgewidth = [4 * d['weight'] for (u, v, d) in G_f.edges(data=True)]
    # edgecolor =[cmap(4 * d['weight']) for (u, v, d) in G_f.edges(data=True)]
    # nx.draw_networkx(G_f, graph_pos, width=np.zeros(N_internal))
    # nx.draw_networkx_edges(G_f, graph_pos, edge_color= edgecolor, width = edgewidth)
    # plt.colorbar(sm, ticks=np.linspace(0,.25, 6))
    #
    #
    # print(KL_score_external(P_f, .3, modular_toy_paper()))
    # print(KL_score(modular_toy_paper(), .3))
    # pk.dump(P_f, open(r"optimal_modular_descendX.pickle", "wb"))

    # G_0 = create_undirected_network(15, 30)
    # bestVal, best = optimize(G_0, .3, 100000, automorphism_count, flipEdge, line_graph= False, minimize= False)
    #
    # edge_labels, inv_labels, output = compute_line_graph_details(best, 30)
    # components, mappings = get_structurally_symmetric(output)
    # print(components)
    #
    # plt.figure(1)
    # G = nx.from_numpy_matrix(best)
    # nx.draw_networkx(G)


    '''
    A_target = modular_toy_paper()
    
    beta_range = np.linspace(1e-6, 2, 500)
    lambda_range = np.linspace(1e-6, 2, 500)
    results = np.zeros((len(beta_range), len(lambda_range)))
    results = pk.load(open(r"beta-lambda-heatmap micro up to 2, multiplicative.pickle", "rb"))
    # for i in range(len(beta_range)):
    #     print(i)
    #     for j in range(len(lambda_range)):
    #         A_init = biased_modular(lambda_range[j])
    #         score = KL_score_external(A_init, beta_range[i], A_target)/KL_score(A_target, beta_range[i])
    #         results[i][j] = score
    '''

    '''
    beta_range = np.linspace(1e-6, 2, 500)
    lambda_cc_range = np.linspace(1e-6, 2, 500)
    lambda_b_range = np.linspace(1e-6, 2, 500)
    # results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))
    # results2 = np.zeros((len(lambda_cc_range), len(lambda_b_range)))
    results = pk.load(open(r"beta-lambda-heatmap micro up to 2, 5, boundary edges ratio.pickle", "rb"))
    # for i in range(len(lambda_cc_range)):
    #     print(i)
    #     for j in range(len(lambda_b_range)):
    #         A_init = biased_modular(lambda_cc_range[i], lambda_b_range[j])
    #         score_ext = KL_score_external(A_init, .01, A_target)
    #         score_baseline = KL_score(A_target, .01)
    #         results[i][j] = score_ext - score_baseline
    #         results2[i][j] = score_ext/score_baseline
    
    # row, col = np.unravel_index(results2.argmin(), results2.shape)
    # results2[row][col] = 100
    
    # plt.figure(1)
    # plt.imshow(results, cmap = 'RdBu',extent=[.01, 2, .01, 2], origin='lower', aspect = 1, norm = mn.MidpointNormalize(midpoint=0))
    # plt.title(r"$D_{KL}(A || f(A^*))-D_{KL}(A || f(A))$", size = 16)
    # plt.xlabel(r"$\lambda_b$", size = 16)
    # plt.ylabel(r"$\lambda _{cc}$", size = 16)
    # cbar = plt.colorbar()
    
    plt.figure(5)
    plt.imshow(results, cmap = 'RdBu',extent=[.01, 2, .01, 2], origin='lower', aspect = 1,vmin = .95, vmax = 1.05, norm = mn.MidpointNormalize(midpoint=1))
    plt.title(r"$\frac{D_{KL}(A || f(A^*))}{D_{KL}(A || f(A))}$", size = 16)
    plt.xlabel(r"$\lambda_b$", size = 16)
    plt.ylabel(r"$\lambda _{cc}$", size = 16)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(['< .96', '.98', '1', '1.02', '> 1.04'])
    
    plt.figure(2)
    for i in range(0, len(lambda_cc_range), 25):
        #plt.ylim([0.6, 2])
        plt.plot(lambda_cc_range, results[:, i], label = r"$\lambda_{b} =$"+str(lambda_b_range[i])[0:4], linewidth = .8, color =
        colorFader('red', 'green', np.power(i/len(lambda_cc_range), .75)))
    plt.xlabel(r"$\lambda_{cc}$", size = 16)
    plt.ylabel(r"$\frac{D_{KL}(A || f(A^*))}{D_{KL}(A || f(A))}$", size = 16)
    plt.legend(prop={'size': 8}, loc = 1, ncol = 2)
    plt.tight_layout()
    
    #minimums
    lambda_b_vals = np.zeros(len(lambda_cc_range) - 1)
    score_vals = np.zeros(len(lambda_cc_range) - 1)
    for i in range(1, len(results)):
        argmin = np.argmin(results[i])
        lambda_b_vals[i-1] = lambda_b_range[argmin]
        score_vals[i-1] = results[i][argmin]
    plt.figure(3)
    plt.plot(lambda_cc_range[1:], score_vals, color = "orange")
    plt.xlabel(r"$\lambda_{cc}$", size = 16)
    plt.ylabel(r"$\frac{D_{KL}(A || f(A^*))}{D_{KL}(A || f(A))}$", size = 16)
    plt.tight_layout()
    
    plt.figure(4)
    plt.plot(lambda_cc_range[1:], lambda_b_vals, color = "orange")
    plt.xlabel(r"$\lambda_{cc}$", size = 16)
    plt.ylabel(r"$\lambda_{b} ^*$", size = 16)
    plt.tight_layout()
    
    # pk.dump(results, open(r"beta-lambda-heatmap micro up to 2, .3 reversed, boundary edges diff.pickle", "wb"))
    # pk.dump(results2, open(r"beta-lambda-heatmap micro up to 2, .3 reversed, boundary edges ratio.pickle", "wb"))
    '''
    plt.show()