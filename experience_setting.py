import networkx as nx
import matplotlib.pyplot as plt
import random

def create_topology(node_num, edge_prob):
    ### Use realistic topology
    # G = nx.read_gml("topology/Aarnet.gml")

    ### Random
    random_seed = 138
    G = nx.gnp_random_graph(node_num, edge_prob, random_seed)
    while nx.is_connected(G) == False:
        G = nx.gnp_random_graph(node_num, edge_prob, random_seed)

    #G = nx.Graph()

    # N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # E = [(0, 1), (0, 4), (0, 6), (0, 7), (1, 9), (2, 3), (2, 5), (2, 8), (2, 9), (3, 4), (3, 6), (6, 7), (7, 8), (7, 9)]
    # G.add_nodes_from(N)
    # G.add_edges_from(E)

    nx.set_node_attributes(G, {n: {'capacity': 2} for n in G.nodes})
    nx.set_node_attributes(G, {n: {'vnf': []} for n in G.nodes})
    nx.set_edge_attributes(G, {e: {'bandwidth': random.randint(10,20)} for e in G.edges})
    nx.set_edge_attributes(G, {e: {'data_rate': 0} for e in G.edges})

    return G

'''
Random the set of clients of multicast task.

clients = (dsts, video_quality)
'''
def create_client(num, node, src, video_type):
    clients = dict()
    
    tmp_N = list(node)
    tmp_N.remove(src)
    dsts = random.sample(tmp_N, num)
    
    for i in range(num):
        clients[dsts[i]] = random.choice(video_type)
    
    return clients

'''
Calculate the bitrate of transocded video. 
'''
def cal_transcode_bitrate(data_rate, input_q, output_q):
    #print(data_rate, input_q, output_q)
    bitrate_table = {'224p': {'224p': 1} \
                    ,'360p': {'224p': 0.7, '360p': 1} \
                    ,'480p': {'224p': 0.5, '360p': 0.7, '480p': 1} \
                    ,'720p': {'224p': 0.3, '360p': 0.5, '480p': 0.7, '720p': 1} \
                    ,'1080p': {'224p': 0.25, '360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}}
    return bitrate_table[input_q][output_q] * data_rate
