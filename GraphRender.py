import copy
from colour import Color
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import GenerateGraph as gg
import GraphLearning as gl
import symmetries as sm
import MidpointNormalize as mn

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

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

def heatmap_render(betaCap, lam1Cap, lam2Cap):
    beta_range = np.linspace(1e-3, betaCap, 500)
    lambda_cc_range = np.linspace(1e-3, lam1Cap, 500)
    lambda_b_range = np.linspace(1e-3, lam2Cap, 500)
    results = np.zeros((len(lambda_cc_range), len(lambda_b_range)))

    network0 = gg.get_lattice_graph([3,5])
    numParams, parameterized = sm.getSymReducedParams(network0, include_nonexistent=False)
    outcomes = np.zeros((len(beta_range), 1))
    bounds = [(1e-6, 20) for i in range(2)]

    for i in range(len(lambda_cc_range)):
        print(i)
        for j in range(len(lambda_b_range)):
            A_init = parameterized([1, lambda_cc_range[i]])
            score_ext = gl.KL_score_external(A_init, beta_range[j], network0)
            score_baseline = gl.KL_score(network0, beta_range[j])
            results[i][j] = score_ext/score_baseline
    plt.figure(5)
    plt.rcParams.update({'font.size': 14})
    cax = plt.imshow(results, cmap = 'RdBu',extent=[.01, 1, .01, 1], origin='lower', vmax = 1.1, vmin = .9, aspect = 1, norm = mn.MidpointNormalize(midpoint=1))
    plt.title(r"$\frac{D_{KL}(A || f(A_{in}))}{D_{KL}(A || f(A))}$", size=18)
    plt.rcParams.update({'font.size': 14})
    plt.xlabel(r"$\beta$", size=20)
    plt.ylabel(r"$\lambda _{l}$", size=20)
    cbar = plt.colorbar(cax, ticks = [.9, .95, 1.0, 1.05, 1.1])
    cbar.ax.set_yticklabels(['<.9','.95','1.0','1.05', '> 1.1'])
    plt.tight_layout()

def get_colors(A):
    all_cols = mcolors.CSS4_COLORS
    G_temp =nx.from_numpy_matrix(A)
    edgecolor = []
    for (u, v, d) in G_temp.edges(data = True):
        edgecolor.append('grey')
       #  if d['weight'] == .2:
       #      edgecolor.append('grey')
       #  if d['weight'] == 1:
       #      edgecolor.append('orange')
       #  if d['weight'] == 2:
       #      edgecolor.append('blue')
       #  if d['weight'] == 4:
       #      edgecolor.append('forestgreen')

        # if d['weight'] == 0.5:
        #     edgecolor.append('grey')
        #    # edgecolor.append('forestgreen')
        #     #edgecolor.append((0, 0.6, 0, 1))
        # else:
        #     edgecolor.append('grey')
        #     #edgecolor.append((0, 0, 0, 1))
    return edgecolor

def render_network(input, fignum, graph_pos = None, k = 1, nodecolors = None):
    plt.figure(fignum)
    G_0 = nx.from_numpy_matrix(input)
    d = dict(G_0.degree)
    if not graph_pos:
        graph_pos = nx.spring_layout(G_0, iterations = 1000, k = k)
    edgewidth = [ d['weight']  for (u, v, d) in G_0.edges(data=True)]
    # edgewidth = [.8 for (u, v, d) in G_0.edges(data=True)]
    #edgecolor = [(0.2, 0.2, 0.2, min(.8  * d['weight'], 1))  for (u, v, d) in G_0.edges(data=True)]
    edgecolor = get_colors(input)
    edgewidth = [max(.5, 1 * abs(d['weight']) ** 1) for (u, v, d) in G_0.edges(data=True)]
    edgecolor = [(0, 0, 0, max(.008, min(.8 * d['weight'], 1))) for (u, v, d) in G_0.edges(data=True)]
    # edgecolor = ['blue' for e in G_0.edges]
    #nx.draw_networkx(G_0, graph_pos, width=np.zeros(len(input)), with_labels= False, node_color = 'lightblue', node_size = [v  * 30 for v in d.values()])
    if nodecolors is None:
        nx.draw_networkx(G_0, graph_pos, width=np.zeros(len(input)), with_labels=False, node_color='lightblue', node_size=50)
    else:
        nx.draw_networkx(G_0, graph_pos, width=np.zeros(len(input)), with_labels=False, node_color=nodecolors, node_size=50)
    nx.draw_networkx_edges(G_0, graph_pos, edge_color=edgecolor, width=edgewidth)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    plt.axis('off')
    return graph_pos
