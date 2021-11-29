'''
Find the multicast path with the minimum cost.

input: 
G' = (N',E') = the copy of G, N = nodes(include M = NFV nodes, F = switches), L = links
S = (s,D,V) = request(service), s = source, D = set of dsts, V = {f1,f2,...} = SFC

output:
Gv = multicast path
'''

import sys
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt

import calculate_function as cal
import Graph
import Client

### Parameters
alpha = 0.6
beta = 1 - alpha
path_cost = sys.float_info.max
instance_num = sys.maxsize
dst_num = random.randint(2, 3)
vnf_num = random.randint(2, 3)
vnf_type = {'a':1,'b':1,'c':1,'d':1,'e':1}
quality_list = ['360p', '480p', '720p', '1080p']

# create_topology()
G = nx.Graph()
N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#N = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
E = [(0, 1), (0, 4), (0, 6), (0, 7), (1, 9), (2, 3), (2, 5), (2, 8), (2, 9), (3, 4), (3, 6), (6, 7), (7, 8), (7, 9)]
#E = [('0', '1'), ('0', '4'), ('0', '6'), ('0', '7'), ('1', '9'), ('2', '3'), ('2', '5'), ('2', '8'), ('2', '9'), ('3', '4'), ('3', '6'), ('6', '7'), ('7', '8'), ('7', '9')]

G.add_nodes_from(N)
G.add_edges_from(E)

nx.set_node_attributes(G, {n: {'capacity': 2} for n in G.nodes})
# the vnf instance can't share with other service
nx.set_node_attributes(G, {n: {'vnf': ''} for n in G.nodes})
nx.set_edge_attributes(G, {e: {'bandwidth': 2} for e in G.edges})
nx.set_edge_attributes(G, {e: {'data_rate': 0} for e in G.edges})

pos = nx.spring_layout(G)

# generate service
src = random.choice(N)
dsts = Client.create_client(dst_num, N, src, quality_list)
dsts.sort(key = lambda d: nx.algorithms.shortest_paths.shortest_path_length(G,src,d[0]))
sfc = random.sample(sorted(vnf_type), vnf_num)
data_rate = 0.2
dst_list = list(d for d,q in dsts)
service = (src, dsts, sfc, data_rate)

for node in N:
    # copy G to G_new, and update the weight of edges
    G_new = copy.deepcopy(G)
    nx.set_edge_attributes(G_new, {e: {'weight': cal.cal_link_weight(G_new,e,alpha,beta,data_rate)} for e in G_new.edges})

    # generate metric closure
    G_metric_closure = nx.algorithms.approximation.metric_closure(G_new,weight='weight')
    edge_labels_Gm = dict([((n1, n2), d['weight']) for n1, n2, d in G_metric_closure.edges(data=True)])

    # generate the set of {src, dsts, n} in G_new
    G_keyNode = nx.Graph()
    G_keyNode.add_nodes_from([src,node])
    G_keyNode.add_nodes_from(dst_list)

    for n1,n2,d in G_metric_closure.edges(data=True):
        if n1 in G_keyNode and n2 in G_keyNode:
            G_keyNode.add_edge(n1,n2,weight=d['distance'])

    # generate MST
    T = nx.minimum_spanning_tree(G_keyNode,weight='distance')

    # replace edges in MST with corresponding paths in G_new
    keynode_path = nx.Graph(T)
    for e in T.edges:
        if e not in G:
            keynode_path.remove_edge(e[0],e[1])
            shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_new, e[0],e[1], weight='weight')
            for j in shortest_path:
                for i in range(len(shortest_path)-1):
                    edge_pair = (shortest_path[i],shortest_path[i+1]) 
                    keynode_path.add_edge(shortest_path[i],shortest_path[i+1],data_rate=0)

    Graph.printGraph(keynode_path, pos, 'keynode_path_'+str(node), service, 'data_rate')
    
    # greedy placement VNF on nodes (for each dst)
    # that its capacity can satisfy the data_rate of VNF
    
    for dst in dsts:
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(multicast_path, src, dst, weight='weight')
        for j in shortest_path:
            if j != src and j != dst:
                for vnf in service[2]:
                    if G.nodes[j]['capacity'] > vnf_type[vnf]:
                        G.nodes[j]['vnf'].append(vnf)

# print Graph
# Graph.printGraph(G, pos, 'G', service, 'bandwidth')
# Graph.printGraph(G_new, pos, 'G_new', service, 'weight')
# Graph.printGraph(G_metric_closure, pos, 'G_metric', service, 'distance')
# Graph.printGraph(G_keyNode, pos, 'G_key', service, 'weight')
# Graph.printGraph(T, pos, 'T', service, 'weight')
