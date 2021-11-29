import matplotlib.pyplot as plt
import random
import copy
import sys
import networkx as nx
from networkx.linalg.algebraicconnectivity import _preprocess_graph

import calculate_function as cal
import Graph
import Client

G = nx.Graph()

N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
E = [(0, 1), (0, 4), (0, 6), (0, 7), (1, 9), (2, 3), (2, 5), (2, 8), (2, 9), (3, 4), (3, 6), (6, 7), (7, 8), (7, 9)]
G.add_nodes_from(N)
G.add_edges_from(E)

nx.set_node_attributes(G, {n: {'capacity': 2} for n in G.nodes})
nx.set_node_attributes(G, {n: {'vnf': []} for n in G.nodes})
nx.set_edge_attributes(G, {e: {'bandwidth': random.randint(2,10)} for e in G.edges})
nx.set_edge_attributes(G, {e: {'data_rate': 0} for e in G.edges})

pos = nx.spring_layout(G)

alpha = 0.6
beta = 1 - alpha
data_rate = 0.2

vnf_num = random.randint(2, 3)
vnf_type = {'a':1,'b':1,'c':1,'d':1,'e':1}

quality_list = ['360p', '480p', '720p', '1080p']

dst_num = random.randint(2, 3)
tree_num = 3
src = random.choice(N)
dsts = Client.create_client(dst_num, N, src, quality_list)
dsts.sort(key = lambda d: nx.algorithms.shortest_paths.shortest_path_length(G,src,d[0]))
sfc = random.sample(sorted(vnf_type), vnf_num)
dst_list = list(d for d,q in dsts)
service = (src, dsts, sfc, data_rate)

G_min = copy.deepcopy(G) # record the min cost network 
multicast_path_min = copy.deepcopy(G) # record the min cost path
index_sfc_min = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list) # record the satisfied vnf of min cost path and place node

for node in N:
    G_new = copy.deepcopy(G)
    nx.set_edge_attributes(G_new, {e: {'weight': cal.cal_link_weight(G_new,e,alpha,beta,data_rate)} for e in G_new.edges})

    G_metric_closure = nx.algorithms.approximation.metric_closure(G_new,weight='weight')

    G_keyNode = nx.Graph()
    G_keyNode.add_nodes_from([src,node])
    G_keyNode.add_nodes_from(dst_list)

    for n1,n2,d in G_metric_closure.edges(data=True):
        if n1 in G_keyNode and n2 in G_keyNode:
            G_keyNode.add_edge(n1,n2,weight=d['distance'])

    # generate MST
    T = nx.minimum_spanning_tree(G_keyNode,weight='distance')

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

    tmp_G = copy.deepcopy(G) # record the temporary capacity and vnf of nodes
    tmp_multicast_path = nx.Graph() # record the temporary multicast path
    index_sfc = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list) # record the satisfied vnf now and place node
    shortest_path_set = {}
    min_path_length = sys.maxsize

    for dst in dst_list:
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(keynode_path, src, dst, weight='weight')
        shortest_path_set[dst] = shortest_path
        if len(shortest_path) < min_path_length:
            min_path_length = len(shortest_path)
        # update nodes of tmp_multicast_path with nodes of shortest_path
        node_attr = {}
        for m in shortest_path:
            if m not in tmp_multicast_path:
                node_attr[m] = {'vnf': []}
            else:
                node_attr[m] = tmp_multicast_path.nodes[m]
        tmp_multicast_path.add_nodes_from(shortest_path)
        nx.set_node_attributes(tmp_multicast_path, node_attr)

        # iterative the node in shortes_path from src to dst
        for i in range(len(shortest_path)-1):
            # update the bandwidth of link
            e = (shortest_path[i],shortest_path[i+1])
            if e not in tmp_multicast_path.edges:
                tmp_G.edges[e]['data_rate'] += data_rate
                tmp_G.edges[e]['bandwidth'] -= data_rate
                tmp_multicast_path.add_edge(*e,data_rate=data_rate)

            # don't place vnf on src and dst
            if shortest_path[i] != src and shortest_path[i] != dst:
                j = shortest_path[i]
                
                # find node to place vnf until finishing sfc
                while index_sfc[dst]['index'] < len(service[2])-1:
                    vnf = service[2][index_sfc[dst]['index']+1]

                    # there are vnf instance on node j
                    if vnf in tmp_G.nodes[j]['vnf']:
                        index_sfc[dst]['index'] += 1
                        index_sfc[dst]['place_node'].append(j)
                        continue
                    
                    # check if node has enough resource to place instance
                    if tmp_G.nodes[j]['capacity'] > vnf_type[vnf]:
                        tmp_G.nodes[j]['vnf'].append(vnf)
                        tmp_multicast_path.nodes[j]['vnf'].append(vnf)
                        tmp_G.nodes[j]['capacity'] -= vnf_type[vnf]
                        index_sfc[dst]['index'] += 1
                        index_sfc[dst]['place_node'].append(j)
                    else:
                        break

        # print('--- ',dst,' ---')
        # print(tmp_multicast_path.nodes.data(True))
        # print(Graph.cal_cost(tmp_G, [tmp_multicast_path], alpha, vnf_type), Graph.count_instance(tmp_multicast_path))
    
    # greedy selection
    if Graph.count_instance(tmp_multicast_path) == Graph.count_instance(multicast_path_min) and \
        Graph.cal_cost(tmp_G, [tmp_multicast_path], alpha, vnf_type) < Graph.cal_cost(G_min, [multicast_path_min], alpha, vnf_type):
            multicast_path_min = copy.deepcopy(tmp_multicast_path)
            G_min = copy.deepcopy(tmp_G)
            index_sfc_min = copy.deepcopy(index_sfc)
    elif Graph.count_instance(tmp_multicast_path) > Graph.count_instance(multicast_path_min):
            multicast_path_min = copy.deepcopy(tmp_multicast_path)
            G_min = copy.deepcopy(tmp_G)
            index_sfc_min = copy.deepcopy(index_sfc)

print('--- environment ---')
print('src = ', src, ", dst = ", dst_list)

# print('--- final ---')
# print(multicast_path_min.nodes(data=True))
# print(Graph.cal_cost(G_min, [multicast_path_min], alpha, vnf_type), Graph.count_instance(tmp_multicast_path))
# print(tmp_multicast_path.edges(data=True))
print("sfc = ", service[2])
print("index_sfc = ", index_sfc_min)
print("shortest path = ", shortest_path_set)

Graph.printGraph(G_min, pos, 'G_min_miss_vnf', service, 'bandwidth')
Graph.printGraph(multicast_path_min, pos, 'path_miss_vnf', service, 'data_rate')

'''
a corrective subroutine that places the missing NF instances 
on the closest NFV node from the multicast topology.
'''
missing_vnf_dsts = []
for dst in index_sfc_min: # find all dst which missing vnf 
    if index_sfc_min[dst]['index'] < len(service[2])-1:
        missing_vnf_dsts.append(dst)

print("miss = ", missing_vnf_dsts)

update_shortest_path_set = copy.deepcopy(shortest_path_set)
# remove unnecessary edges: the path from last vnf node to dst
for dst in missing_vnf_dsts:
    last_place_node = index_sfc_min[dst]['place_node'][-1] if index_sfc_min[dst]['index'] > -1 else -1
    last_node_index = shortest_path_set[dst].index(last_place_node) if last_place_node != -1 else 0
    del update_shortest_path_set[dst][last_node_index+1:]

update_edge = set()
for dst in update_shortest_path_set:
    for i in range(len(update_shortest_path_set[dst])-1):
        e = (update_shortest_path_set[dst][i], update_shortest_path_set[dst][i+1])
        update_edge.add(e)

for dst in missing_vnf_dsts:
    for i in range(len(shortest_path_set[dst])-1):
        e = (shortest_path_set[dst][i],shortest_path_set[dst][i+1])
        if e not in update_edge:
            G_min.edges[e]['bandwidth'] += data_rate
            G_min.edges[e]['data_rate'] -= data_rate
            multicast_path_min.remove_edge(*e)

multicast_path_remove = copy.deepcopy(multicast_path_min)

Graph.printGraph(G, pos, 'G', service, 'bandwidth')
Graph.printGraph(G_min, pos, 'G_min_remove', service, 'bandwidth')
Graph.printGraph(multicast_path_min, pos, 'path_remove', service, 'data_rate')

print("update shortest path = ", update_shortest_path_set)

# place vnf nearest the common path
for dst in missing_vnf_dsts:
    find_distance = 0
    place_flag = 0
    while index_sfc_min[dst]['index'] < len(service[2])-1:
        last_node = update_shortest_path_set[dst][-1]
        if place_flag == 0:
            find_distance += 1
        else:
            find_distance = 1
            place_flag = 0
        alternate_nodes_len = nx.single_source_shortest_path_length(G, last_node, find_distance)
        alternate_nodes = list(n for n in nx.single_source_shortest_path_length(G, last_node, find_distance))
        # print("alt_node = ",alternate_nodes_len)
        for i in alternate_nodes:
            if place_flag == 1:
                break
            # if find_distance > 1, the node that distance = 1 have been searched
            # it don't have to consider the node again
            if find_distance > 1 and alternate_nodes_len[i] == find_distance-1:
                continue
            
            last_node = update_shortest_path_set[dst][-1]
            shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, i, weight='data_rate')
            
            vnf = service[2][index_sfc_min[dst]['index']+1]
            if i == last_node or i == src or i == dst:
                continue
                
            # there are vnf instance on node i
            while vnf in G_min.nodes[i]['vnf']:
                index_sfc_min[dst]['index'] += 1
                index_sfc_min[dst]['place_node'].append(i)
                place_flag = 1

                if find_distance == 1:
                    update_shortest_path_set[dst].append(i)
                    if i not in multicast_path_min:
                        multicast_path_min.add_node(i, vnf=[])
                    e = (last_node, i)
                    # if previous_node to i don't have path then build it
                    Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
                else:
                    for j in range(len(shortest_path)-1):
                        update_shortest_path_set[dst].append(shortest_path[j+1])
                        if j not in multicast_path_min:
                            multicast_path_min.add_node(shortest_path[j], vnf=[])
                        e = (shortest_path[j], shortest_path[j+1])
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)

                if index_sfc_min[dst]['index'] == len(service[2])-1: # finish sfc
                    break
                vnf = service[2][index_sfc_min[dst]['index']+1]
                last_node = i
            
            if index_sfc_min[dst]['index'] == len(service[2])-1: # finish sfc
                break
                    
            # check if node has enough resource to place instance
            while G_min.nodes[i]['capacity'] > vnf_type[vnf]:
                G_min.nodes[i]['vnf'].append(vnf)
                if i not in multicast_path_min:
                    multicast_path_min.add_node(i, vnf=[])
                multicast_path_min.nodes[i]['vnf'].append(vnf)
                G_min.nodes[i]['capacity'] -= vnf_type[vnf]
                index_sfc_min[dst]['index'] += 1
                index_sfc_min[dst]['place_node'].append(i)
                place_flag = 1

                if find_distance == 1:
                    update_shortest_path_set[dst].append(i)
                    e = (last_node, i)
                    # if previous_node to i don't have path then build it
                    Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
                else:
                    for j in range(len(shortest_path)-1):
                        update_shortest_path_set[dst].append(shortest_path[j+1])
                        e = (shortest_path[j], shortest_path[j+1])
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
                
                if index_sfc_min[dst]['index'] == len(service[2])-1: # finish sfc
                    break
                vnf = service[2][index_sfc_min[dst]['index']+1]
                last_node = i
            
            if index_sfc_min[dst]['index'] == len(service[2])-1: # finish sfc
                break

print("final update shortest path = ", update_shortest_path_set)
Graph.printGraph(multicast_path_min, pos, 'tmp', service, 'data_rate')

# construct the path from the last_placement_node to dst
for dst in missing_vnf_dsts:
    last_node = update_shortest_path_set[dst][-1]
    shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, dst, weight='data_rate')
    # print("shortest path from last_node to dst = ", shortest_path)
    for i,j in enumerate(shortest_path):
        if i == 0:
            continue
        if j not in multicast_path_min:
            multicast_path_min.add_node(j, vnf=[])
        e = (last_node, j)
        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
        update_shortest_path_set[dst].append(j)
        last_node = j

print("final_path", update_shortest_path_set)
Graph.printGraph(G_min, pos, 'G_min_update', service, 'bandwidth')
Graph.printGraph(multicast_path_min, pos, 'path_update', service, 'data_rate')

Graph.printGraph(G_new, pos, 'G_new', service, 'weight')
Graph.printGraph(G_metric_closure, pos, 'G_metric', service, 'distance')
Graph.printGraph(G_keyNode, pos, 'G_key', service, 'weight')
Graph.printGraph(T, pos, 'T', service, 'weight')

'''
Check whether each path in multicast_path satisfy the link transmission rate requirement.
If a single-path solution is infeasible, it will be extended to the multipath solution
'''
G_final = copy.deepcopy(G)
for m in multicast_path_min:
    G_final.nodes[m]['vnf'] = multicast_path_min.nodes[m]['vnf']
path_final = copy.deepcopy(multicast_path_min)
path_final.remove_edges_from(multicast_path_min.edges)

use_tree = []

for dst in update_shortest_path_set:
    dst_path = update_shortest_path_set[dst]
    for i in range(len(dst_path)-1):
        success_flag = False
        tree_list = list(nx.all_simple_paths(G, source=dst_path[i], target=dst_path[i+1]))
        tree_list_num = len(tree_list)
        
        # sort all candidate paths in a descending order 
        # based on the amount of residual transmission resource.
        tree_resource = dict()
        for k in range(tree_list_num):
            tree_resource[k] = 0
            for m in range(len(tree_list[k])-1):
                tree_resource[k] += G.edges[tree_list[k][m],tree_list[k][m+1]]['bandwidth']
        tree_resource_sort = sorted(tree_resource.items(), key=lambda x:x[1], reverse=True)
        
        for j in range(tree_num-1):
            tmp_index = min(j, tree_list_num)
            tmp_path = tree_list[tree_resource_sort[tmp_index][0]]
            #print("tmp_path = ",tmp_path)
            min_bandwidth = sys.maxsize
            for m in range(len(tmp_path)-1):
                if G_final.edges[tmp_path[m],tmp_path[m+1]]['bandwidth'] < min_bandwidth:
                    min_bandwidth = G_final.edges[tmp_path[m],tmp_path[m+1]]['bandwidth']
                
            # check if edges of tmp_path have enough resource to transmission data
            if min_bandwidth > data_rate:
                use_tree.append(tmp_path)
                for m in tmp_path:
                    if m not in path_final:
                        path_final.add_node(m, vnf=[])
                for m in range(len(tmp_path)-1):
                    e = (tmp_path[m],tmp_path[m+1])
                    allocate_data_rate = data_rate / (tmp_index+1) # averagely allocate data rate to each tree
                    #print(allocate_data_rate, ", tmp_index = " ,tmp_index)
                    Graph.add_new_edge(G_final, path_final, update_shortest_path_set, dst, e, allocate_data_rate)
                success_flag = True
                break
        '''
        if success_flag == True:
            return path_final
        '''
print(use_tree)
Graph.printGraph(G_final, pos, 'G_final', service, 'bandwidth')
Graph.printGraph(path_final, pos, 'path_final', service, 'data_rate')