import os
import pickle as pk
from copy import deepcopy
import bct
import copy
import mynumpy as np
import pydot
import pandas as pd
import networkx as nx
import scipy as sp
from scipy import optimize as op
from scipy import stats
from scipy.stats import binned_statistic
import symmetries as sm
import GenerateGraph as gg
import GraphRender as gr
import matplotlib.pyplot as plt
import matplotlib.ticker
import ProcessData as pr
import graphviz
os.environ["PATH"] += os.pathsep + 'C:/Users/billy/Downloads/graphviz-2.38/release/bin'
import sys
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.seterr(divide='ignore')

rng = np.random.RandomState()
#seeded_rng = np.random.RandomState(17310145)
seeded_rng = rng

head_dir = "C:/Users/billy/PycharmProjects/GraphLearningAgents/"
#head_dir = "/data/jux/bqqian/GraphLearning/"

# languages = head_dir + "graphs_Language_share/"
# music = head_dir + "graphs_Music_share/"
# web = head_dir + "graphs_Web_share/"
# social = head_dir + "graphs_Social_share/"
# citation = head_dir + "graphs_Citation_share/"
# semantic = head_dir + "graphs_Semantic_share/"
textbooks = head_dir + "textbooks3/"

def learn(A, beta):
    A = normalize(A)
    inverse_argument = np.identity(len(A)) - np.exp(-beta)*A
    inverse = sp.linalg.inv(inverse_argument)
    return normalize((1-np.exp(-beta))*(A @ inverse))
def learn_to_undirected(A, beta, normalizer):
    out = learn(A, beta)
    out = unnormalize(out)
    return normalizer * out / np.sum(out)

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

def getNumEdges(A): #Assumes undirected input!
    return np.count_nonzero(A) / 2

def get_stationary3(A): #only applies for unnormalized weighted graph inputs
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

KLCount = 0
def KL_Divergence(U, V, weighted_net = None):
    global KLCount
    U = normalize(U)
    V = normalize(V)
    if weighted_net is None:
        pi = get_stationary2(U)
    else:
        pi = get_stationary3(weighted_net)
    combined = np.einsum('i, ij -> ij', pi, U)
    logged = np.log(V/U)
    logged[U == 0] = 0
    result = combined.T @ logged
    outcome = -np.trace(result)
    # KLCount += 1
    # if KLCount % 1000 == 0:
    #     print(outcome, KLCount)
    return outcome

def KL_score(A, beta, A_target = None):
    return KL_Divergence(A, learn(A, beta))

def KL_score_ext_zipped(input, beta):
    cc_bias, b_bias = input
    if cc_bias < 1e-8 or b_bias < 1e-8:
        return 100
    return KL_score_external(gg.biased_modular(cc_bias, b_bias), beta, gg.modular_toy_paper())

def KL_score_external(A_input, beta, A_target, weighted_net = None):
    return KL_Divergence(A_target, learn(A_input, beta), weighted_net = weighted_net)



def get_KL_ext_general(A_target, include_nonexistent = True):
    numParams, parameterized = sm.getSymReducedParams(A_target, include_nonexistent = include_nonexistent)
    def cost_func(input, P_0, pi, beta, J, I):
        return KL_score_external(parameterized(input), beta, A_target)
    return numParams, cost_func, parameterized

def get_KL_ext_mod(parameterized):
    def cost(cc_weight, beta, A_target):
        return KL_score_external(parameterized(cc_weight), beta, A_target)
    return cost

def get_pickleable_params(A, include_nonexistent = True, force_unique = False):
    comps, comp_maps, edge_labels, inv_labels = sm.unique_edges(A, force_unique = force_unique)
    if include_nonexistent:
        A_c = sm.get_complement_graph(A)
        comps_c, comp_maps_c, edge_labels_c, inv_labels_c = sm.unique_edges(A_c, force_unique = force_unique)
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

def one_param_cost_func(input, parameterized, beta, A_target):
    B = parameterized(input)
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

def uniformity_cost(P_0, A, beta):
    learned = learn(A, beta)
    terms = learned[P_0 > 0].flatten()
    diffs = np.subtract.outer(terms, terms)
    return np.sum(diffs * diffs)

def uniformity_cost_zipped(input, N_tot, N_comms, beta):
    A = gg.modular_toys_general(N_tot, N_comms, input[0], input[1])
    return uniformity_cost(gg.modular_toys_general(N_tot, N_comms, 1, 1), A, beta)

def KL_modular_toys_general(input, N_tot, N_comms, beta):
    cc_bias, b_bias = input
    return KL_score_external(gg.modular_toys_general(N_tot, N_comms, cc_bias, b_bias), beta, gg.modular_toys_general(N_tot, N_comms, 1, 1))

def frobenius_norm_cost(P_0, A, beta):
    learned = learn(A, beta)
    return sp.linalg.norm(normalize(P_0) - learned)

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
                tri_count[(e[0], e[1])] += A[c[0]][c[1]] * A[c[0]][c[2]] * A[c[1]][c[2]]
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

def get_optimal_directly(A_target, beta):
    I = np.identity(len(A_target))
    inv_argument = I*(1-np.exp(-beta)) + np.exp(-beta)*A_target
    # print(np.linalg.cond(inv_argument, p='fro'))
    inv = sp.linalg.inv(inv_argument)
    return inv @ A_target

def optimize_learnability(network0, weighted, symmInfo, parameterized, beta, include_nonexistent, KL = True, get_weights = False):
    numParams, comps, comps_c, inv_labels, inv_labels_c = symmInfo
    bounds = [(0, 1) for i in range(numParams)]
    outcome = op.dual_annealing(pickleable_cost_func, bounds=bounds,
        args=(comps, comps_c, inv_labels, inv_labels_c, beta, network0, include_nonexistent, KL, weighted),
                                                                accept = -40, maxiter = 1000, maxfun= 1.2e6)
    A = parameterized(outcome.x)
    score_original = KL_score(network0, beta)
    score = KL_score_external(A, beta, network0)
    if get_weights:
        return A, score_original, score, outcome.x
    return A, score_original, score

def optimize_one_param_learnability(network0, parameterized, beta):
    bounds = [(0, 100)]
    outcome = op.dual_annealing(one_param_cost_func, bounds = bounds,
                                args = (parameterized, beta, network0), accept = -10, maxiter=1000, maxfun = 1e6)
    A = parameterized(outcome.x[0])
    score_original = KL_score(network0, beta)
    score = KL_score_external(A, beta, network0)
    return A, outcome.x[0], score_original, score

def WS_trials(N_tot, k, p_res, trials, beta):
    with np.errstate(invalid='ignore'):
        p_vals = np.logspace(0, 4, p_res)
        p_vals /= p_vals[-1]
        print(p_vals)
        opts = np.zeros(p_res)
        scores_orig =  np.zeros(p_res)
        scores_s = np.zeros(p_res)
        for i in range(p_res):
            for j in range(trials):
                network, parameterized = gg.small_world_parameterized(N_tot, k, p_vals[i])
                network_s, opt, score_orig, score_s = optimize_one_param_learnability(network, parameterized, beta)
                opts[i] += opt
                scores_orig[i] += score_orig
                scores_s[i] += score_s
        opts /= trials
        scores_orig /= trials
        scores_s /= trials
        return p_vals, opts, scores_orig, scores_s

def SBM_trials(N_tot, N_comm, edges, alpha, frac_res, trials, beta):
    with np.errstate(invalid='ignore'):
        frac_modules = np.linspace(.2, 1, frac_res, endpoint = False)
        mod_opts, hMod_opts = np.zeros(frac_res), np.zeros(frac_res)
        scores_mod_orig, scores_hMod_orig = np.zeros(frac_res), np.zeros(frac_res)
        scores_mod_s, scores_hMod_s = np.zeros(frac_res), np.zeros(frac_res)
        for i in range(frac_res):
            for j in range(trials):
                mod, parMod = gg.get_random_modular(N_tot, N_comm, edges, frac_modules[i])
                hMod, parhMod = gg.get_hierarchical_modular(N_tot, N_comm, edges, frac_modules[i], alpha)
                mod_s, mod_opt, score_mod_orig, score_mod_s = optimize_one_param_learnability(mod, parMod, beta)
                hMod_s, hMod_opt, score_hMod_orig, score_hMod_s = optimize_one_param_learnability(hMod, parhMod, beta)
                mod_opts[i] += mod_opt
                hMod_opts[i] += hMod_opt
                scores_mod_orig[i] += score_mod_orig
                scores_hMod_orig[i] += score_hMod_orig
                scores_mod_s[i] += score_mod_s
                scores_hMod_s[i] += score_hMod_s
        mod_opts /= trials
        hMod_opts /= trials
        scores_mod_orig /= trials
        scores_hMod_orig /= trials
        scores_mod_s /= trials
        scores_hMod_s /= trials
        return frac_modules, mod_opts, hMod_opts, scores_mod_orig, scores_hMod_orig, scores_mod_s, scores_hMod_s
def all_core_periphery_avged(networks_orig, networks_opt):
    output = np.zeros((15, 4))
    output_std = np.zeros((15, 4))
    classified = [[[], [], [], []] for _ in range(15)]
    for j in range(10):
        classifications, per_comm_assignments, _, _ = core_periphery_analysis(networks_orig[j])
        for i in range(15):
            classified_vals, _ = classify_vals(classifications, networks_orig[j], networks_opt[j][i], per_comm_assignments)
            for k in range(len(classified_vals)):
                classified[i][k].extend(classified_vals[k])
    for i in range(15):
        for k in range(4):
            output[i][k] = np.mean(classified[i][k])
            output_std[i][k] = np.std(classified[i][k])
    return output, output_std, classified
def all_core_periphery(networks_orig, networks_opt):
    output = np.zeros((10, 15, 4))
    output_std = np.zeros((10, 15, 4))
    for j in range(10):
        classifications, per_comm_assignments, _, _ = core_periphery_analysis(networks_orig[j])
        for i in range(15):
            classified_vals, _ = classify_vals(classifications, networks_orig[j], networks_opt[j][i], per_comm_assignments)
            for k in range(len(classified_vals)):
                output[j][i][k] = np.mean(classified_vals[k])
                output_std[j][i][k] = np.std(classified_vals[k])
    return output, output_std
def core_periphery_analysis(network0):
    network0 /= np.sum(network0)
    C, Q_core = bct.core_periphery_dir(network0)
    per_nodes = []
    for i in range(len(C)):
        if C[i] == 0:
            per_nodes.append(i)
    G = nx.from_numpy_matrix(network0)
    G_per = G.subgraph(per_nodes)
    per_network = np.array(nx.to_numpy_matrix(G_per))
    M_per, Q_comm_per = bct.community_louvain(per_network)
    print(Q_comm_per, "Q")
    # print(M_per, Q_comm_per)
    per_comm_assignments = {}
    for i in range(len(per_nodes)):
        per_comm_assignments[per_nodes[i]] = M_per[i]
    classifications = [[], [], []] # index 0 means periphery-periphery edge, 1 means periphery-core, 2 means core-core
    for i in range(len(network0) - 1):
        for j in range(i+1, len(network0)):
            if network0[i][j] > 0:
                classifications[C[i] + C[j]].append((i, j))
    return classifications, per_comm_assignments, G_per, M_per

def classify_vals(classifications, network0, network_opt, per_comm_assignments):
    classified_vals = [[], [], [],
                       []]  # periphery-periphery intramod, periphery-periphery intermod, periphery-core, core-core
    classified_edges = [[], [], [], []]
    edge_factors = get_edge_values(network0, network_opt)
    for j in range(len(classifications[0])):
        e0, e1 = classifications[0][j]
        if per_comm_assignments[e0] == per_comm_assignments[e1]:
            classified_vals[0].append(edge_factors[classifications[0][j]])
            classified_edges[0].append(classifications[0][j])
        else:
            classified_vals[1].append(edge_factors[classifications[0][j]])
            classified_edges[1].append(classifications[0][j])
    for i in range(1, 3):
        for j in range(len(classifications[i])):
            classified_vals[i + 1].append(edge_factors[classifications[i][j]])
            classified_edges[i + 1].append(classifications[i][j])
    return classified_vals, classified_edges

def get_diff_stats(network0, network_opt):
    edge_vals = []
    tri_participation = []
    betweenness = []
    edge_degrees = []
    network0 /= np.sum(network0)
    network_opt /= np.sum(network_opt)
    edges, tri_count, total_triangles = compute_triangle_participation(network0)
    edge_factors = get_edge_values(network0, network_opt)
    betweenness_dict = nx.centrality.edge_betweenness_centrality(nx.from_numpy_matrix(network0), weight = 'weight')
    for i in range(len(edges)):
        # if edge_factors[(edges[i][0], edges[i][1])] > 0:
        edge_degrees.append(.5 * (np.sum(network0[edges[i][0]]) + np.sum(network0[edges[i][1]])))
        edge_vals.append(edge_factors[(edges[i][0], edges[i][1])])
        tri_participation.append(tri_count[(edges[i][0], edges[i][1])])
        betweenness.append(betweenness_dict[(edges[i][0], edges[i][1])])

    # classifications, per_comm_assignments, G_per, M_per = core_periphery_analysis(network0)
    # classified_vals, classified_edges = classify_vals(classifications, network0, network_opt, per_comm_assignments)
    #
    # # A = np.zeros((len(network0), len(network0)))
    # # for i in range(4):
    # #     if i == 0:
    # #         val = .2
    # #     if i == 1:
    # #         val = 1
    # #     if i == 2:
    # #         val = 2
    # #     if i == 3:
    # #         val = 4
    # #     for j in range(len(classified_edges[i])):
    # #         e0, e1 = classified_edges[i][j]
    # #         A[e0][e1], A[e1][e0] = val, val
    # # print("IM RENDERING BRO")
    # # gr.render_network(A, 10)
    # # X, Y = bct.grid_communities(M_per)
    # # print(X, Y)
    # per_network = nx.to_numpy_matrix(G_per)
    # per_network /= np.sum(per_network)
    # layout_mask = np.zeros((len(per_network), len(per_network)))
    # for i in range(len(layout_mask)):
    #     for j in range(len(layout_mask)):
    #         if M_per[i] == M_per[j] and i != j:
    #             layout_mask[i][j] = 1
    #
    # graph_pos = nx.spring_layout(nx.from_numpy_matrix(layout_mask), k = .45)
    # gr.render_network(per_network,11, graph_pos = graph_pos, nodecolors= .25 * np.array(M_per))
    # plt.figure(200, figsize = (4.4, 3.6))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel("P-P within-cluster weight scaling", fontsize = 14)
    # plt.ylabel("Probability density")
    # plt.rcParams.update({'font.size': 16})
    # plt.hist(classified_vals[0], bins=30, density = True, color = "tomato", linewidth = .6, edgecolor='black')
    # plt.tight_layout()
    # plt.figure(201, figsize = (4.4, 3.6))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel("P-P cross-cluster weight scaling", fontsize = 14)
    # plt.ylabel("Probability density")
    # plt.rcParams.update({'font.size': 16})
    # plt.hist(classified_vals[1], bins=30, density = True, color = "lightgreen", linewidth = .6,  edgecolor='black')
    # plt.tight_layout()
    # plt.figure(202, figsize = (4.4, 3.6))
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel("P-C weight scaling")
    # plt.ylabel("Probability density")
    # plt.rcParams.update({'font.size': 16})
    # plt.hist(classified_vals[2], bins=30, density = True, color = "cornflowerblue", linewidth = .6,  edgecolor='black')
    # plt.tight_layout()
    # plt.figure(203, figsize = (4.4, 3.6))
    # plt.xlabel("C-C weight scaling")
    # plt.ylabel("Probability density")
    # plt.hist(classified_vals[3], bins=30, density=True, color="grey", linewidth=.6, edgecolor='black')
    # plt.tight_layout()

    # #binned_vals, dividers, _ = binned_statistic(tri_participation, edge_vals, 'mean', bins=20)
    # #dividers = dividers[:-1]
    # plt.figure(5)
    # plt.rcParams.update({'font.size': 16})
    # #plt.scatter(dividers, binned_vals)
    # plt.scatter(tri_participation, edge_vals, s=10, alpha=.1)
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel("Edge clustering coefficient")
    # plt.ylabel("Optimal edge scaling")
    #
    # # gradient, intercept, r_value, p_value, std_err = stats.linregress(tri_participation, edge_vals)
    # # print(r_value, p_value)
    # # mn = np.amin(tri_participation)
    # # mx = np.amax(tri_participation)
    # # x1 = np.linspace(mn, mx, 500)
    # # y1 = gradient * x1 + intercept
    # # plt.plot(x1, y1, '-r')
    #
    # # binned_vals, dividers, _ = binned_statistic(betweenness, edge_vals, 'mean', bins=10)
    # # dividers = dividers[:-1]
    # plt.figure(6)
    # plt.rcParams.update({'font.size': 16})
    # # plt.scatter(dividers, binned_vals)
    # plt.scatter(betweenness, edge_vals, s=10, alpha=.1)
    # plt.rcParams.update({'font.size': 16})
    # plt.xlabel('Edge betweenness centrality')
    # plt.ylabel("Optimal edge scaling")
    #
    # gradient, intercept, r_value, p_value, std_err = stats.linregress(betweenness, edge_vals)
    # print(r_value, p_value)
    # mn = np.amin(betweenness)
    # mx = np.amax(betweenness)
    # x1 = np.linspace(mn, mx, 500)
    # y1 = gradient * x1 + intercept
    # plt.plot(x1, y1, '-r')

    data = list(zip(tri_participation, edge_vals))
    return data
    # data = list(zip(tri_participation, edge_vals))
    # data.sort()
    # print(data)
    # bin_size = 300
    # new_pairs = []
    # stdev_pairs = []
    # j = 0
    # while j < len(data):
    #     count, x, y = 0, [], []
    #     while count < bin_size and j + count < len(data):
    #         x.append(data[j + count][0])
    #         y.append(data[j + count][1])
    #         count += 1
    #     if count < bin_size:
    #         break
    #     new_pairs.append((np.mean(x), np.mean(y)))
    #     stdev_pairs.append((np.std(x), np.std(y)))
    #     j += count
    # new_pairs = np.array(new_pairs)
    # stdev_pairs = np.array(stdev_pairs)
    # plt.figure(7)
    # #plt.scatter(new_pairs[:, 0], new_pairs[:, 1])
    # #plt.errorbar(new_pairs[:, 0], new_pairs[:, 1], xerr = stdev_pairs[:,0], yerr=stdev_pairs[:,1], fmt='o', capsize = 2, elinewidth= .5)

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

if __name__ == '__main__':
    betas = np.linspace(1e-3, .2, 15)
    beta_index = 14
    textbook_index = 8
    #arg_1 = 0
    # arg_1 = int(sys.argv[1]) - 1 # from 0 to 149
    # beta_index = arg_1 % len(betas)
    # textbook_index = arg_1 // 15
    beta = betas[beta_index]

  #   A_0 = gg.regularized_sierpinski(3,5)
  #   symInfo = get_pickleable_params(A_0, include_nonexistent= False, force_unique= False)
  #   numParams, parameterized = sm.getSymReducedParams(A_0, include_nonexistent=False, force_unique=False)
  #   betas2 = np.linspace(1e-3, 1, 150)
  #   # outcomes = np.zeros((len(betas2), numParams))
  #   # scores = np.zeros((len(betas2), 2))
  #   # for i in range(len(betas2)):
  #   #     A, score_original, score, weights = optimize_learnability(A_0, A_0, symInfo, parameterized, betas2[i],
  #   #                                                  include_nonexistent=False, get_weights = True)
  #   #     outcomes[i] = weights
  #   #     outcomes[i] /= weights[0]
  #   #     scores[i][0] = score_original
  #   #     scores[i][1] = score
  #   #     print(i, score_original, score, outcomes[i])
  #   # pk.dump([betas, outcomes, scores], open("betas_outcomes_scores_sierpinski_150.pk", "wb"))
  #   betas, outcomes, scores = pk.load(open("betas_outcomes_scores_sierpinski_150.pk", "rb"))
  #   plt.figure(figsize = (5.5,4.5))
  #   plt.rcParams.update({'font.size': 16})
  # #  plt.plot(betas2, outcomes[:, 0], color = "grey", linewidth = .7)
  #   plt.plot(betas2, outcomes[:, 1], color = "orange", linewidth = 1, label = r'$\lambda _{cc}^2$')
  # #  plt.plot(betas2, outcomes[:, 2], color = "blue", linewidth = .7)
  #   plt.plot(betas2, outcomes[:, 3], color = "forestgreen", linewidth = 1, label = r'$\lambda _{b}^2$')
  #   plt.legend(frameon = False)
  #   plt.rcParams.update({'font.size': 16})
  #   plt.xlabel(r'$\beta$')
  #   plt.ylabel('Optimal level-2 weights')
  #   plt.tight_layout()
  #
  #   plt.figure(figsize = (5.5,4.5))
  #   ax = plt.gca()
  #   plt.rcParams.update({'font.size': 16})
  #   plt.plot(betas2, outcomes[:,3] - outcomes[:,0], color = "firebrick", label = r'$\lambda _{b}^2 - \lambda _{b}^3$')
  #   plt.plot(betas2, outcomes[:,1] - outcomes[:,2],  color = "cadetblue", label = r'$\lambda_{cc}^2 - \lambda _{cc}^3$')
  #   plt.legend(frameon = False)
  #   plt.rcParams.update({'font.size': 16})
  #   plt.xlabel(r'$\beta$')
  #   plt.ylabel('Optimal cross-level weight diff.')
  #   plt.ticklabel_format(axis = 'y', style = 'sci')
  #   ax.yaxis.major.formatter.set_powerlimits((0, 0))
  #   ax.yaxis.major.formatter._useMathText = True
  #   plt.tight_layout()
  #
  #   plt.figure(figsize = (5.5,4.5))
  #   plt.rcParams.update({'font.size': 16})
  #   plt.plot(betas2, scores[:, 0], color = "lightgrey", label='Original ('+r'$A_{in} = A$'+')')
  #   plt.plot(betas2, scores[:, 1], color = "black", label='Optimized (' + r'$A_{in} = A^{*}$' + ')')
  #   plt.rcParams.update({'font.size': 16})
  #   plt.xlabel(r'$\beta$')
  #   plt.ylabel('KL Divergence, ' + r'$D_{KL}(A||f(A_{in}))$')
  #   plt.legend(frameon = False)
  #   plt.tight_layout()
  #
  #   #A_0 = parameterized([0.2,1,2,4])
  #   # graph_pos = nx.drawing.nx_pydot.graphviz_layout(nx.from_numpy_matrix(A_0), prog = 'sfdp')
  #   # graph_pos = nx.kamada_kawai_layout(nx.from_numpy_matrix(A_0))
  #   # graph_pos = nx.spring_layout(nx.from_numpy_matrix(A_0))
  #   graph_pos = pr.process_node_pos("sierpinski.txt.cyjs", 243)
  #   gr.render_network(A_0, 25, graph_pos = graph_pos)
  #   learned = unnormalize(learn(A_0, .2))
  #   learned /= np.sum(learned)
  #   learned *= np.sum(A_0)
  #   gr.render_network(learned, 26, graph_pos = graph_pos)
  #
  #   A, _, _ = optimize_learnability(A_0, A_0, symInfo, parameterized, .2, include_nonexistent=False)
  #   A /= np.sum(A)
  #   A *= np.sum(A_0)
  #   gr.render_network(A, 27, graph_pos = graph_pos)
  #   learned = unnormalize(learn(A, beta))
  #   learned /= np.sum(learned)
  #   learned *= np.sum(A_0)
  #   gr.render_network(learned, 28, graph_pos=graph_pos)
  #   plt.show()


    indices = np.load(textbooks+"all_index.npy", allow_pickle= True)
    networks_orig = np.load(textbooks + "cooc_mats.npy", allow_pickle= True)
    for i in range(len(networks_orig)):
        for j in range(len(networks_orig[i])):
            networks_orig[i][j][j] = 0
    for i in range(len(networks_orig)):
        networks_orig[i] /= np.sum(networks_orig[i])
        #networks_orig[i] = normalize(networks_orig[i])


    networks = []
    scores = []
    for i in range(10):
        networks.append(np.load(textbooks + str(i)+"_opt_networks.npy", allow_pickle= True))
        scores.append(np.load(textbooks + str(i)+"_KL.npy", allow_pickle= True))
    for i in range(len(networks)):
        for j in range(len(betas)):
            networks[i][j] /= np.sum(networks[i][j])

    markers = ["o", "+", "*", "D", "x", "d", "^", "s", "v", ">"]
    colors = ["orange", "sienna", "limegreen", "deepskyblue", "steelblue", "purple", "lightseagreen", "darkgrey", "black", "red"]
    names = ["Treil", "Axler", "Edwards", "Lang", "Petersen", "Robbiano", "Bretscher", "Greub", "Hefferson", "Strang"]

    np.random.shuffle(colors)
    plt.figure(100, figsize=(6.5, 4.5))
    plt.rcParams.update({'font.size': 16})
    plt.xlabel(r'$\beta$')
    plt.ylabel('KL Divergence Ratio')
    plt.rcParams.update({'font.size': 16})
    for i in [0,1,2,3,4,5,6,7,8,9]:
        # plt.ylim([0.25, 1.7])
        plt.scatter(betas, scores[i][:, 0]/ scores[i][:, 1], s = 30, alpha = .7, color = colors[i], marker = markers[i], label = names[i])
        plt.plot(betas, scores[i][:, 0]/ scores[i][:, 1], linewidth = .6, color=colors[i])
        plt.legend(frameon=False, prop={'size': 12}, labelspacing=.2, handletextpad=0, borderpad = 0, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    network0 = networks_orig[textbook_index]
    network_opt = networks[textbook_index][beta_index]
    #get_diff_stats(network0, network_opt)

    data = []
    for i in range(10):
        print(i)
        data_curr = get_diff_stats(networks_orig[i], networks[i][14])
        data.extend(data_curr)
    data.sort()
    data = np.array(data)

    bin_size = 500
    new_pairs = []
    stdev_pairs = []
    j = 0
    # while data[j][0] == 0:
    #     j += 1
    while j < len(data):
        count, x, y = 0, [], []
        while count < bin_size and j + count < len(data):
            x.append(data[j + count][0])
            y.append(data[j + count][1])
            count += 1
        if count < bin_size:
            break
        new_pairs.append((np.mean(x), np.mean(y)))
        stdev_pairs.append((np.std(x), np.std(y)))
        j += count
    new_pairs = np.array(new_pairs)
    stdev_pairs = np.array(stdev_pairs)
    plt.figure(7, figsize = (5.5, 4.5))
    ax = plt.gca()
    plt.scatter(new_pairs[:, 0], new_pairs[:, 1], color = 'sandybrown', s = 30)
    plt.xlabel('Edge degree centrality', fontsize = 16)
    plt.ylabel('Optimal weight scaling')
    #plt.ylim([-0.1,3.25])
    plt.ticklabel_format(axis='x', style='sci')
    ax.xaxis.major.formatter.set_powerlimits((0, 0))
    ax.xaxis.major.formatter._useMathText = True
    plt.tight_layout()
    #plt.errorbar(new_pairs[:, 0], new_pairs[:, 1], xerr = stdev_pairs[:,0], yerr=stdev_pairs[:,1], fmt='o', capsize = 2, elinewidth= .5)


    CPData, stdev = all_core_periphery(networks_orig, networks)
    colors_class = ["tomato", "lightgreen", "cornflowerblue", "grey"]
    class_names = ["P-P within-cluster", "P-P cross-cluster", "P-C", "C-C"]
    for i in range(10):
        plt.figure(300 + i)
        for k in range(4):
            plt.scatter(betas, CPData[i, :, k], s = 50, alpha = .7, color = colors_class[k], label = class_names[k], marker = '*')
            plt.plot(betas, CPData[i, :, k], linewidth = .6, color = colors_class[k])
            #plt.errorbar(betas, CPData[i, :, k], yerr = stdev[i,:, k], capsize= 5, ecolor = colors[k])
            #plt.fill_between(betas, CPData[i, :, k] - stdev[i, :, k], CPData[i, :, k] + stdev[i, :, k], alpha = .2)
            plt.legend(frameon=False)


    CPData_all, stdev_all, classified = all_core_periphery_avged(networks_orig, networks)
    colors2 = ["tomato", "lightgreen", "cornflowerblue", "grey"]
    labels = [r'$\langle\lambda _{P,P}^{wc}\rangle$', r'$\langle\lambda _{P,P}^{cc}\rangle$', r'$\langle\lambda _{P,C}\rangle$', r'$\langle\lambda _{C,C}\rangle$']
    #labels = [r'$\lambda _{P,P}^{wc}$', r'$\lambda _{P,P}^{cc}$', r'$\lambda _{P,C}$', r'$\lambda _{C,C}$']
    #labels = ['P-P within-cluster', 'P-P cross-cluster', 'P-C', 'C-C']

    plt.figure(310, figsize = (5.5, 4.5))
    plt.rcParams.update({'font.size': 16})
    plt.xlabel(r'$\beta$')
    plt.ylabel('Mean optimal weight scaling')
    plt.rcParams.update({'font.size': 16})
    ax = plt.gca()
    for k in range(4):
        plt.scatter(betas, CPData_all[:, k], s=50, alpha=.7, color=colors2[k], label=labels[k], marker = '*')
        plt.plot(betas, CPData_all[:, k], linewidth=1, color=colors2[k])
        #plt.fill_between(betas, CPData_all[:, k] - stdev_all[:, k], CPData_all[:, k] + stdev_all[:, k], alpha=.1, color = colors[k])
        plt.legend(frameon = False, loc = 'upper right',  labelspacing=.2, handletextpad=0, borderpad = 0)
        #plt.legend(frameon = False, ncol=2)
    plt.tight_layout()
    # plt.yscale('log')

    # handles, labels = ax.get_legend_handles_labels()
    # plt.close()
    # fig_legend = plt.figure(figsize=(2, 2))
    # plt.box(False)
    # axi = fig_legend.add_subplot(111)
    # fig_legend.legend(handles, labels, loc='center', scatterpoints=1)
    # axi.xaxis.set_visible(False)
    # axi.yaxis.set_visible(False)
    # fig_legend.canvas.draw()
    # fig_legend.show()

    plt.figure(200, figsize = (3.25, 2.7))
    plt.rcParams.update({'font.size': 16})
    #plt.xlabel("P-P within-cluster weight scaling", fontsize = 14)
    plt.xlabel(r'$\lambda _{P,P}^{wc}$', fontsize = 20)
    plt.ylabel("Prob. density")
    plt.rcParams.update({'font.size': 16})
    plt.hist(classified[14][0], bins=30, density = True, color = colors2[0], linewidth = .6, edgecolor='black')
    plt.tight_layout()
    plt.figure(201, figsize = (3.25, 2.7))
    plt.rcParams.update({'font.size': 16})
    #plt.xlabel("P-P cross-cluster weight scaling", fontsize = 14)
    plt.xlabel(r'$\lambda _{P,P}^{cc}$', fontsize = 20)
    plt.ylabel("Prob. density")
    plt.rcParams.update({'font.size': 16})
    plt.hist(classified[14][1], bins=30, density = True, color = colors2[1], linewidth = .6,  edgecolor='black')
    plt.tight_layout()
    plt.figure(202, figsize = (3.25, 2.7))
    plt.rcParams.update({'font.size': 16})
    #plt.xlabel("P-C weight scaling")
    plt.xlabel(r'$\lambda _{P,C}$', fontsize = 20)
    plt.ylabel("Prob. density")
    plt.rcParams.update({'font.size': 16})
    plt.hist(classified[14][2], bins=30, density = True, color = colors2[2], linewidth = .6,  edgecolor='black')
    plt.tight_layout()
    plt.figure(203, figsize = (3.0, 2.7))
    #plt.xlabel("C-C weight scaling")
    plt.xlabel(r'$\lambda _{C,C}$', fontsize = 20)
    plt.ylabel("Prob. density")
    plt.hist(classified[14][3], bins=30, density=True, color=colors2[3], linewidth=.6, edgecolor='black')
    plt.tight_layout()

    plt.show()

    # #network0 = np.load(textbooks + "cooc_mats.npy", allow_pickle= True)
    # network0 = np.load("cooc_mats.npy", allow_pickle=True)
    # network0 = network0[textbook_index]
    # for i in range(len(network0)):
    #     network0[i][i] = 0
    # A_0 = normalize(network0)
    #
    # unweighted = deepcopy(network0)
    # unweighted[unweighted > 0] = 1
    #
    # symInfo = get_pickleable_params(unweighted, include_nonexistent= False, force_unique= True)
    # numParams, parameterized = sm.getSymReducedParams(unweighted, include_nonexistent=False, force_unique= True)
    #
    # A, score_original, score = optimize_learnability(A_0, network0, symInfo, parameterized, beta, include_nonexistent= False)
    #
    # f = open(head_dir + str(textbook_index) + "_opt_networks.npy", "rb")
    # f_metrics = open(head_dir + str(textbook_index) + "_KL.npy", "rb")
    #
    # networks = np.load(f, allow_pickle= True)
    # metrics = np.load(f_metrics, allow_pickle= True)
    #
    # f.close()
    # f_metrics.close()
    #
    # networks[beta_index] = A
    # metrics[beta_index] = [score, score_original]
    #
    # f = open(head_dir + str(textbook_index) + "_opt_networks.npy", "wb")
    # f_metrics = open(head_dir + str(textbook_index) + "_KL.npy", "wb")
    #
    # np.save(f, networks)
    # np.save(f_metrics, metrics)
    #
    # f.close()
    # f_metrics.close()








