import copy

import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import os
import tempfile
import itertools
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
    inverse_argument = np.identity(len(A)) - np.exp(-beta)*A
    inverse = np.linalg.pinv(inverse_argument)
    return (1-np.exp(-beta))*np.dot(A, inverse)

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
    pi = get_stationary2(U)
    result = 0
    for i in range(len(U)):
        for j in range(len(U)):
            if not np.isclose(U[i][j], 0, rtol = 1e-8) and not np.isclose(V[i][j], 0, rtol = 1e-8):
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
    return KL_Divergence(normalize(A), normalize(learn(normalize(A), beta)))

def KL_score_external(A_input, beta, A_target):
    return KL_Divergence(normalize(A_target), normalize(learn(A_input, beta)))

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
        curr = compose(flipFunc, seeded_rng.randint(len(A)))(best)
        count = 0
        while not isConnected(curr):
            curr = compose(flipFunc, seeded_rng.randint(len(A)))(best)
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

G = nx.random_regular_graph(4, N_internal)
A = np.array(nx.adjacency_matrix(G).todense(), dtype = float)
bestVal, bestVals, unnormalized_network  = optimize(A, .33, 20000, KL_score, rewire_regular, minimize = True)
IG = ig.Graph.Adjacency(unnormalized_network.tolist())
G_internal = nx.from_numpy_matrix(unnormalized_network)
graph_pos = nx.spring_layout(G_internal, iterations=50)
nx.draw_networkx(G_internal, graph_pos)
print(str(55)+"\t"+str(bestVal)+"\t"+str(nx.transitivity(G_internal))+"\t"+str(IG.count_automorphisms_vf2()))

plt.figure()
plt.xlabel("iteration")
plt.ylabel("KL Divergence between original network and learned network")
plt.plot(bestVals)
'''
clustering = np.zeros(200000)
symmetry = np.zeros(200000)
learnability = np.zeros(200000)
for i in range(200000):
    G = nx.random_regular_graph(4, N_internal)
    A = np.array(nx.adjacency_matrix(G).todense(), dtype = float)
    IG = ig.Graph.Adjacency(A.tolist())
    score = KL_Divergence(normalize(A), normalize(learn(normalize(A), .33)))
    symmetry[i] = IG.count_automorphisms_vf2()
    learnability[i] = score
    clustering[i] = nx.transitivity(G)
    #graph_pos=nx.spring_layout(G, k=1.0, iterations=50)
    #nx.draw_networkx(G, graph_pos)
    print(str(i)+"\t"+str(clustering[i])+"\t"+str(score)+"\t"+str(symmetry[i]))
plt.figure()
plt.scatter(clustering, learnability, s = 10, alpha = .2)
plt.figure()
plt.scatter(symmetry, learnability, s = 10, alpha = .2)
plt.figure()
plt.scatter(symmetry, clustering, s= 10, alpha = .2)

plt.figure()
modular = modular_toy_paper()
G_modular = nx.from_numpy_matrix(modular)
graph_pos=nx.spring_layout(G_modular, k=1.0, iterations=50)
nx.draw_networkx(G_modular, graph_pos)
print(str(nx.transitivity(G_modular))+"\t"+str(KL_Divergence(normalize(modular), normalize(learn(normalize(modular), .3)))))
'''
'''
bestVal, bestVals, network = optimize_divergence(create_undirected_network(N_internal, 10), .3, 2000, minimize = True)
#network = modular_toy_paper()
letters = ["y","u","i","o","n"]
rd.shuffle(letters)
generate_key_sequence(network, letters, 100)
'''
'''
plt.figure(2)
plt.title("agent network")
agent_network =  normalize(create_agent_network(N, sources, 30))
#agent_network = normalize(create_undirected_network(N, 30))

G = nx.from_numpy_matrix(agent_network, create_using= nx.MultiDiGraph)
graph_pos=nx.spring_layout(G, k=1.0, iterations=50)
nx.draw_networkx(G, graph_pos)

print("AGENT NETWORK")
print(pd.DataFrame(normalize(agent_network)))

plt.figure()
agent_networks_init = np.zeros((N, N_internal, N_internal), dtype = float)
unnormalized_network = modular_toy_paper()

plt.figure(55)
#bestVal, bestVals, unnormalized_network = optimize_divergence(create_undirected_network(N_internal, 30), .5, 10000, minimize = True)
unnormalized_network = create_undirected_network(N_internal, 100)
print("DONE")
bestVal = KL_Divergence(normalize(unnormalized_network), normalize(learn(normalize(unnormalized_network), .33)))
G = ig.Graph.Adjacency(unnormalized_network.tolist())
G_internal = nx.from_numpy_matrix(unnormalized_network)
graph_pos = nx.spring_layout(G_internal, iterations=50)
nx.draw_networkx(G_internal, graph_pos)
print(str(55)+"\t"+str(bestVal)+"\t"+str(nx.transitivity(G_internal))+"\t"+str(G.count_automorphisms_vf2()))


plt.figure()
#unnormalized_network = flipEdge(N_internal, modular_toy_paper(), ensureConnected = True)
unnormalized_network = modular_toy_paper()
# unnormalized_network[4][5] = 0
# unnormalized_network[5][4] = 0
# unnormalized_network[10][14] = 1
# unnormalized_network[14][10] = 1
G = ig.Graph.Adjacency(unnormalized_network.tolist())
G_internal = nx.from_numpy_matrix(unnormalized_network)
graph_pos = nx.circular_layout(G_internal)
nx.draw_networkx(G_internal, graph_pos)
learnability = KL_Divergence(normalize(unnormalized_network), normalize(learn(normalize(unnormalized_network), .5)))
transitivity = nx.transitivity(G_internal)
symmetry = G.count_automorphisms_vf2()
print(str(learnability)+"\t"+str(transitivity)+"\t"+str(symmetry))
'''
'''
'''
'''
agent_networks_init[0] = normalize(unnormalized_network)
for i in range(1, sources):
    #agent_networks_init[i] = normalize(flipEdge(N_internal, unnormalized_network, ensureConnected = True))
    agent_networks_init[i] = normalize(unnormalized_network)


#agent_networks_init[2] = normalize(create_undirected_network(N_internal, 15))

for i in np.arange(0, N, 5):
    plt.figure(i+11)
    G_internal = nx.from_numpy_matrix(agent_networks_init[i], create_using= nx.MultiDiGraph)
    graph_pos=nx.spring_layout(G_internal, iterations = 50)
    edgewidth = [ d['weight'] for (u,v,d) in G_internal.edges(data=True)]

    nx.draw_networkx(G_internal, graph_pos, width = np.zeros(N_internal))
    nx.draw_networkx_edges(G_internal, graph_pos, width = edgewidth,)

betas = np.ones(N)*.5

agent_networks, divergences = agent_network_sim(agent_network, agent_networks_init, 100, betas)

for i in np.arange(N):
    plt.figure(37 + i)
    G_internal2 = nx.from_numpy_matrix(normalize(agent_networks[i]), create_using= nx.MultiDiGraph)
    graph_pos=nx.spring_layout(G_internal2, iterations = 50)
    edgewidth = [ d['weight'] for (u,v,d) in G_internal2.edges(data=True)]
    nx.draw_networkx(G_internal2, graph_pos, width = np.zeros(N_internal))
    nx.draw_networkx_edges(G_internal2, graph_pos, width = edgewidth,)


plt.figure(3)
plt.xlabel("Iteration")
plt.ylabel("KL Divergence with Source Network")
for i in range(N):
    plt.plot(divergences[i], label = "agent "+str(i))
plt.legend(loc = 'center right')
'''
'''
internal = create_undirected_network(N_internal, 30)
vals_list = []
for i in range(30):
    bestVal, bestVals, bestNetwork = optimize_divergence(create_undirected_network(N_internal, 30), .3, 5000, minimize= True)
    vals_list.append(bestVals)

plt.figure(7)
G_internal = nx.from_numpy_matrix(bestNetwork)
graph_pos=nx.spring_layout(G_internal, k=1.0, iterations=50)
nx.draw_networkx(G_internal, graph_pos)

plt.figure(6)
for i in range(30):
    plt.plot(vals_list[i], label = str(i), linewidth = .5)
plt.legend()
print(pd.DataFrame(normalize(bestNetwork)))
print(pd.DataFrame(normalize(learn(bestNetwork, .3))))

transitivity_optimized = nx.transitivity(G_internal)
print(transitivity)
print(transitivity_optimized)
'''
# network = normalize(np.array(get_random_modular(N, 1, 12, 0), dtype = float))
# print(pd.DataFrame(network))
#
# U, V, divergences, divergences_init, divergences_init2 = two_agents(network, 500, .8, .3)
#
# print(pd.DataFrame(U))
# print(pd.DataFrame(V))
# plt.figure(0)
# plt.plot(divergences)
# plt.figure(1)
# plt.title("phase plot")
# plt.scatter(divergences_init, divergences_init2, s = 15)

plt.show()