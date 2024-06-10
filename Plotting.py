import math
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import networkx as nx

__all__ = ['draw_level_scheme', 
            'draw_transition_scheme']

def _align_close_levels(g, pos):
    tol = np.max(g.nodes(data='energy')) / 20
    for n in g.nodes():
        in_edges = g.in_edges(n)
        for e in in_edges:
            src = e[0]
            en = g.nodes()[n]['energy']
            en_src = g.nodes()[src]['energy']
            if np.abs(en - en_src) < tol:
                pos[src][0] = pos[n][0]


def _align_single_edges(g, pos):
    # If a node only has one outgoing edge, position
    # the origin node directly above the destination node,
    # as long as the energy between them is not too small
    energy_frac = 0.1
    for n in g.nodes():
        if g.out_degree(n) == 1:
            nchild = list(g.out_edges(n))[0][1]
            pos[n][0] = pos[nchild][0]
            # if ((g.nodes()[n]['energy'] - g.nodes()[nchild]['energy'])/EMax > energy_frac):
            #     pos[n][0] = pos[nchild][0]


# Produce a shift value to move a node's position
def _node_offset(value, xscale):
    if xscale == 0:
        scl = 1
    else:
        scl = xscale

    return rnd.choice([-1,1]) * rnd.uniform(low=0.1*scl, high=0.3*scl)


def _avoid_vertical_skips(g, pos):
    # Try to avoid vertical edges crossing nodes they don't intersect
    xcoords = [p[1][0] for p in pos.items()]

    tol = 5
    for position in pos.items():
        x = position[1][0]

        # list of vertical edges along this x-coordinate
        v_edges = [e for e in g.edges() if pos[e[0]][0] == x and pos[e[1]][0] == x]

        for ve in v_edges:
            src = ve[0]
            des = ve[1]
            for ve_other in v_edges:
                if ve == ve_other:
                    continue
                    
                step = max(xcoords)*rnd.uniform(low=0.05, high=0.3)

                xcoords = [p[1][0] for p in pos.items()]
                xscale = np.max(xcoords) - np.min(xcoords)

                src_other = ve_other[0]
                des_other = ve_other[1]
                if (src_other == src):
                    pos[des_other][0] += _node_offset(pos[des_other][0], xscale)
                    break
                elif (des_other == des):
                    pos[src_other][0] += _node_offset(pos[src_other][0], xscale)
                    break
                elif (src_other > src and des_other < src):
                    pos[des_other][0] += _node_offset(pos[des_other][0], xscale)
                    break
                elif (src_other > des and des_other < des):
                    pos[des_other][0] += _node_offset(pos[des_other][0], xscale)
                    break

            v_edges = [e for e in g.edges() if pos[e[0]][0] == x and pos[e[1]][0] == x]


def _get_node_labels(g, pos):
    node_label_min_frac = 0.05
    node_label_min_length = node_label_min_frac
    node_labels = dict(g.nodes(data='energy'))
    
    xcoords = [p[1][0] for p in pos.items()]
    ycoords = [p[1][1] for p in pos.items()]
    xextent = np.max(xcoords) - np.min(xcoords)
    yextent = np.max(ycoords) - np.min(ycoords)
    
    if (xextent == 0): xextent = 1
    if (yextent == 0): yextent = 1

    for n1 in g.nodes():
        for n2 in g.nodes():
            # normalize difference because x and y axes can 
            # have vastly different scales
            dx = (pos[n1][0] - pos[n2][0]) / xextent
            dy = (pos[n1][1] - pos[n2][1]) / yextent
            dist = np.sqrt(dx*dx + dy*dy)
            
            if(n1 > n2):
                if dist < node_label_min_length:
                    if(n2 in node_labels):
                        node_labels.pop(n2)
        
    return node_labels


def _get_edge_labels(g, pos):
    xcoords = [p[1][0] for p in pos.items()]
    ycoords = [p[1][1] for p in pos.items()]
    xextent = np.max(xcoords) - np.min(xcoords)
    yextent = np.max(ycoords) - np.min(ycoords)
    
    if (xextent == 0): xextent = 1
    if (yextent == 0): yextent = 1

    edge_label_min_frac = 0.15
    edge_label_min_length = edge_label_min_frac
    edge_labels = {}
    for n1 in g.nodes():
        for n2 in g.nodes():
            # normalize difference because x and y axes can 
            # have vastly different scales
            dx = (pos[n1][0] - pos[n2][0]) / xextent
            dy = (pos[n1][1] - pos[n2][1]) / yextent
            dist = np.sqrt(dx*dx + dy*dy)
            
            if(g.has_edge(n1, n2)):            
                # Only add edge label if distance is not small
                if dist >= edge_label_min_length:
                    edge_labels.update({(n1,n2): f'{g[n1][n2]["weight"]:.2f}'})

    # Remove edge weight label if there is only one out_edge
    # i.e. the weight is 1.0
    for (u,v,w) in g.edges(data='weight'):
        if w == 1.0:
            if (u,v) in edge_labels:
                edge_labels.pop((u,v))
                
    return edge_labels


def _draw_level_symbols(g, pos):
    Nl = g.number_of_nodes()
    hline_frac = 1.0/Nl * 1.5
    xcoords = [p[1][0] for p in pos.items()]
    xextent = np.max(xcoords) - np.min(xcoords)
    hline_dist = hline_frac*xextent

    for n in g.nodes():
        enj = g.nodes(data='energy')[n]
        xlj = pos[n][0] - hline_dist/2
        xhj = pos[n][0] + hline_dist/2
        plt.hlines(y=enj, xmin=xlj, xmax=xhj, color='black', zorder=3)


def draw_level_scheme(ls, ax, color_map=['black'], branch=False):
    g = ls.g
    emax = ls.emax

    energy_bin_frac = 1/10
    
    for i, layer in enumerate(nx.topological_generations(g)):
        for n in layer:
            g.nodes[n]["layer"] = i

    layer_size = emax * energy_bin_frac
    for n, en in g.nodes(data='energy'):
        g.nodes[n]['energy_layer'] = math.ceil(en / layer_size)

    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")
    for k in pos:
        pos[k][1] *= -1

    # Change vertical position to energy level
    for k in pos:
        pos[k][1] = g.nodes[k]['energy']
        
    _align_single_edges(g, pos)
    _align_close_levels(g, pos)
    _avoid_vertical_skips(g, pos)
    n_labels = _get_node_labels(g, pos)
    e_labels = _get_edge_labels(g, pos)

    # weight_labels = {(u,v): f'{l:.2f}' for u, v, l in g.edges(data='weight')}
    # node_labels = {u: en for u, en in g.nodes(data='energy')}
    n_sizes = [len(str(en)) * 50 for u, en in g.nodes(data='energy')]
    # n_sizes = [g.in_degree(u) * 350 for u in g.nodes()]
    
    nx.draw_networkx(g, pos=pos, with_labels=False, node_shape = "_", node_size=n_sizes)
    nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_shape="", node_size=n_sizes, alpha=0.4)
    nx.draw_networkx_edges(g, ax=ax, pos=pos, node_size=n_sizes, nodelist=list(g.nodes()), edgelist=list(g.edges()), edge_color=color_map)
    nx.draw_networkx_labels(g, pos, ax=ax, labels=n_labels, verticalalignment='bottom', horizontalalignment='right', bbox=dict({'color':'white', 'alpha':0.5}))
    _draw_level_symbols(g, pos)
    
    if(branch):
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=e_labels, label_pos=0.8)
    # nx.draw_networkx_node_labels(g, pos, node_labels=node_labels)


def draw_transition_scheme(ts, ax, branch=False):
    g = ts.g

    energy_bin_frac = 1/10
    
    for i, layer in enumerate(nx.topological_generations(g)):
        for n in layer:
            g.nodes[n]["layer"] = i

#     layer_size = emax * energy_bin_frac
#     for n, en in g.nodes(data='energy'):
#         g.nodes[n]['energy_layer'] = math.ceil(en / layer_size)
# #         print(n, en, round(en / layer_size))

    pos = nx.multipartite_layout(g, subset_key="layer", align="horizontal")
    for k in pos:
        pos[k][1] *= -1

    # # Change vertical position to energy level
    # for k in pos:
    #     pos[k][1] = g.nodes[k]['energy']
        
    _align_single_edges(g, pos)
    # _align_close_levels(g, pos)
    _avoid_vertical_skips(g, pos) 
    # n_labels = _get_node_labels(g, pos)
    e_labels = _get_edge_labels(g, pos)

    # weight_labels = {(u,v): f'{l:.2f}' for u, v, l in g.edges(data='weight')}
    # node_labels = {u: en for u, en in g.nodes(data='energy')}
    # n_sizes = [len(str(en)) * 50 for u, en in g.nodes(data='energy')]
    # n_sizes = [g.in_degree(u) * 350 for u in g.nodes()]
    
    # nx.draw_networkx(g, pos=pos, with_labels=False, node_shape = "_", node_size=n_sizes)
    nx.draw_networkx_nodes(g, ax=ax, pos=pos, node_shape="s", alpha=0.4)
    nx.draw_networkx_edges(g, ax=ax, pos=pos, nodelist=list(g.nodes()), edgelist=list(g.edges()))
    # nx.draw_networkx_labels(g, pos, ax=ax, labels=n_labels, verticalalignment='bottom', horizontalalignment='right', bbox=dict({'color':'white', 'alpha':0.5}))
    # _draw_level_symbols(g, pos)
    
    if(branch):
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=e_labels, label_pos=0.8)
    # nx.draw_networkx_node_labels(g, pos, node_labels=node_labels)