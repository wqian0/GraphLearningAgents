import copy
from colour import Color
import numpy as np
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
import random as rd
import msvcrt
from scipy import stats
os.environ["PATH"] += os.pathsep + 'C:/Users/billy/Downloads/graphviz-2.38/release/bin'
import small_world as sw
import igraph as ig
import MidpointNormalize as mn
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
    if delLoops:
        for i in range(len(A)):
            B[i][i] = 0.0
    for i in range(len(A)):
        if not np.all(B[i] == 0):
            B[i] /= np.sum(B[i])
    return B

def KL_Divergence(U, V):
    U = normalize(U)
    V = normalize(V)
    pi = get_stationary2(U)
    result = 0
    for i in range(len(U)):
        for j in range(len(U)):
            if not np.isclose(U[i][j], 0, rtol = 1e-10) and not np.isclose(V[i][j], 0, rtol = 1e-10):
                result += pi[i] * U[i][j] * np.log(V[i][j]/U[i][j])
    return -result

def two_agents(init, iterations, beta_1, beta_2):
    U = init
    V = np.zeros((N_internal, N_internal))
    divergences = np.zeros(iterations)
    divergences_init = np.zeros(iterations)
    divergences_init2 = np.zeros(iterations)
    for i in range(iterations):
        print(i)
        V = V + learn(U, beta_2)
        V = normalize(V)

        U = U + learn(V, beta_1)
        U = normalize(U)

        divergences[i] = KL_Divergence(U, V)
        divergences_init[i] = KL_Divergence(init, U)
        divergences_init2[i] = KL_Divergence(init, V)

    return U, V, divergences, divergences_init, divergences_init2

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

def printMatrixToFile(M, file):
    for r in range(len(M)):
        for c in range(len(M[0])):
            file.write(str(M[r][c]) + "\t")
        file.write("\n")
    file.write("\n")

def KL_score(A, beta, A_target = None):
    return KL_Divergence(A, learn(A, beta))

def KL_score_external(A_input, beta, A_target):
    return KL_Divergence(A_target, learn(A_input, beta))

def automorphism_count(A, beta, A_target = None):
    IG = ig.Graph.Adjacency(A.tolist())
    return IG.count_automorphisms_vf2()

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
def optimize(A, beta, iterations, scoreFunc, flipFunc, minimize = True, A_target = None):
    bestVal = float('inf')
    bestVals = []
    collecting = False
    curr = deepcopy(A)
    best = deepcopy(A)
    factor = 1
    if not minimize:
        factor = -1
    for i in range(iterations):
        currScore = factor * scoreFunc(curr, beta, A_target)
        if currScore < bestVal and isConnected(curr):
            bestVal = currScore
            best = deepcopy(curr)
            collecting = True
        if collecting:
            bestVals.append(bestVal)
        curr = compose(flipFunc, 1)(best)
        count = 0
        while not isConnected(curr):
            curr = compose(flipFunc, 1)(best)
            count += 1
            if count > 30:
                return bestVal, factor * np.array(bestVals), best
        print(str(i)+"\t"+str(currScore))
    print(str(bestVal)+"\t BEST VAL!")
    return bestVal, factor * np.array(bestVals), best

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


A_target = modular_toy_paper()

beta_range = np.linspace(1e-6, 2, 500)
lambda_cc_range = np.linspace(1e-6, 2, 500)
lambda_b_range = np.linspace(1e-6, 2, 500)
results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))
results2 = np.zeros((len(lambda_cc_range), len(lambda_b_range)))
results = pk.load(open(r"beta-lambda-heatmap micro up to 2, .1, boundary edges diff.pickle", "rb"))
results2 = pk.load(open(r"beta-lambda-heatmap micro up to 2, .1, boundary edges ratio.pickle", "rb"))
# for i in range(len(lambda_cc_range)):
#     print(i)
#     for j in range(len(lambda_b_range)):
#         A_init = biased_modular(lambda_cc_range[i], lambda_b_range[j])
#         score_ext = KL_score_external(A_init, 5, A_target)
#         score_baseline = KL_score(A_target, 5)
#         results[i][j] = score_ext - score_baseline
#         results2[i][j] = score_ext/score_baseline

# row, col = np.unravel_index(results2.argmin(), results2.shape)
# results2[row][col] = 100
plt.figure(1)
plt.imshow(results, cmap = 'RdBu',extent=[.01, 2, .01, 2], origin='lower', aspect = 1, norm = mn.MidpointNormalize(midpoint=0))
plt.title(r"$D_{KL}(A || f(A^*))-D_{KL}(A || f(A))$", size = 16)
plt.xlabel(r"$\lambda_b$", size = 16)
plt.ylabel(r"$\lambda _{cc}$", size = 16)
plt.colorbar()

plt.figure(5)
plt.imshow(results2, cmap = 'RdBu',extent=[.01, 2, .01, 2], origin='lower', vmax = 1.3, vmin = .8, aspect = 1, norm = mn.MidpointNormalize(midpoint=1))
plt.title(r"$\frac{D_{KL}(A || f(A^*))}{D_{KL}(A || f(A))}$", size = 16)
plt.xlabel(r"$\lambda_b$", size = 16)
plt.ylabel(r"$\lambda _{cc}$", size = 16)
plt.colorbar()

plt.figure(2)
for i in range(0, len(lambda_cc_range), 25):
    #plt.ylim([0.6, 2])
    plt.plot(lambda_cc_range, results2[:, i], label = r"$\lambda_{b} =$"+str(lambda_b_range[i])[0:4], linewidth = .8, color =
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


plt.show()