import os
import pickle as pk
from copy import deepcopy
import copy
import mynumpy as np
import pandas as pd
import networkx as nx
import scipy as sp
from scipy import optimize as op
from scipy import stats
import symmetries as sm
import GenerateGraph as gg
import GraphRender as gr
import matplotlib.pyplot as plt
import sys
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

np.seterr(divide='ignore')

rng = np.random.RandomState()
#seeded_rng = np.random.RandomState(17310145)
seeded_rng = rng

#head_dir = "C:/Users/billy/PycharmProjects/GraphLearningAgents/"
head_dir = "/data/jux/bqqian/GraphLearning/"

# languages = head_dir + "graphs_Language_share/"
# music = head_dir + "graphs_Music_share/"
# web = head_dir + "graphs_Web_share/"
# social = head_dir + "graphs_Social_share/"
# citation = head_dir + "graphs_Citation_share/"
# semantic = head_dir + "graphs_Semantic_share/"
# textbooks = head_dir + "textbooks/"

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

def get_optimal_directly(A_target, beta):
    I = np.identity(len(A_target))
    inv_argument = I*(1-np.exp(-beta)) + np.exp(-beta)*A_target
    # print(np.linalg.cond(inv_argument, p='fro'))
    inv = sp.linalg.inv(inv_argument)
    return inv @ A_target

def optimize_learnability(network0, weighted, symmInfo, parameterized, beta, include_nonexistent, KL = True):
    numParams, comps, comps_c, inv_labels, inv_labels_c = symmInfo
    bounds = [(0, 1) for i in range(numParams)]
    outcome = op.dual_annealing(pickleable_cost_func, bounds=bounds,
        args=(comps, comps_c, inv_labels, inv_labels_c, beta, network0, include_nonexistent, KL, weighted),
                                                                accept = -10, maxiter = 1000, maxfun= 1e6)
    A = parameterized(outcome.x)
    score_original = KL_score(network0, beta)
    score = KL_score_external(A, beta, network0)
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


if __name__ == '__main__':
    betas = np.linspace(1e-3, .2, 15)
    arg_1 = int(sys.argv[1]) - 1 # from 0 to 149

    beta_index = arg_1 % len(betas)
    beta = betas[beta_index]
    textbook_index = arg_1 // 15

    #network0 = np.load(textbooks + "cooc_mats.npy", allow_pickle= True)
    network0 = np.load("cooc_mats.npy", allow_pickle=True)
    network0 = network0[textbook_index]
    for i in range(len(network0)):
        network0[i][i] = 0
    A_0 = normalize(network0)

    unweighted = deepcopy(network0)
    unweighted[unweighted > 0] = 1

    symInfo = get_pickleable_params(unweighted, include_nonexistent= False, force_unique= True)
    numParams, parameterized = sm.getSymReducedParams(unweighted, include_nonexistent=False, force_unique= True)

    A, score_original, score = optimize_learnability(A_0, network0, symInfo, parameterized, beta, include_nonexistent= False)

    f = open(head_dir + str(textbook_index) + "_opt_networks.npy", "rb")
    f_metrics = open(head_dir + str(textbook_index) + "_KL.npy", "rb")

    networks = np.load(f, allow_pickle= True)
    metrics = np.load(f_metrics, allow_pickle= True)

    f.close()
    f_metrics.close()

    networks[beta_index] = A
    metrics[beta_index] = [score, score_original]

    f = open(head_dir + str(textbook_index) + "_opt_networks.npy", "wb")
    f_metrics = open(head_dir + str(textbook_index) + "_KL.npy", "wb")

    np.save(f, networks)
    np.save(f_metrics, metrics)

    f.close()
    f_metrics.close()








