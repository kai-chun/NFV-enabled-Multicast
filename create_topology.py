import matplotlib.pyplot as plt
import networkx as nx
import random

def create_topology(node_num, edge_prob):
    ### Use realistic topology
    # G = nx.read_gml("topology/Aarnet.gml")

    ### Random
    # random_seed = 138
    # G = nx.gnp_random_graph(node_num, edge_prob, random_seed)

    G = nx.Graph()

    N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    E = [(0, 1), (0, 4), (0, 6), (0, 7), (1, 9), (2, 3), (2, 5), (2, 8), (2, 9), (3, 4), (3, 6), (6, 7), (7, 8), (7, 9)]
    G.add_nodes_from(N)
    G.add_edges_from(E)

    nx.set_edge_attributes(G, {e: {'weight': random.randint(1, 10)} for e in G.edges})

    pos=nx.spring_layout(G)
    edge_labels = dict([((n1, n2), d['weight']) for n1, n2, d in G.edges(data=True)])

    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.show()
    # G.node[node_name]['index'] = G.nodes().index(node_name)
    # print(G.edges)
    # print(G.nodes)