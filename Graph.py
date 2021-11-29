import matplotlib.pyplot as plt
import random
import copy
import networkx as nx
import numpy

import calculate_function as cal

def printGraph(G, pos, filename, service, weight):
    color_node = []
    dst_list = list(d for d,q in service[1])
    for n in G:
        if n == service[0]:
            color_node.append('#E3432B')
        elif n in dst_list:
            color_node.append('#FFD448')  
        else:
            color_node.append('gray')

    edge_labels = dict([((n1, n2), round(d[weight],3)) for n1, n2, d in G.edges(data=True)])
    
    plt.figure(1)
    if filename in ['tmp', 'G_min_miss_vnf', 'path_miss_vnf', 'G_min_remove', 'path_remove', 'G_min_update', 'path_update', 'G_final', 'path_final']: 
        node_labels = dict([(n, d['vnf']) for (n,d) in G.nodes(data=True)])
        pos_nodes = dict((node,(coords[0], coords[1]-0.05)) for node, coords in pos.items())
        nx.draw_networkx_labels(G, pos=pos_nodes, labels=node_labels, font_color='blue', font_size=8)
        color_edge = []
        for n1,n2,d in G.edges(data=True):
            if d['data_rate'] != 0:
                color_edge.append('g')
            else:
                color_edge.append('k') 
        nx.draw(G, pos, node_color=color_node, edge_color=color_edge, with_labels=True)
    else:
        nx.draw(G, pos, node_color=color_node, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.savefig('img/'+filename+'.png')
    plt.close()

'''
calculate the cost of multicast path

total cost = transmission cost + processing cost
'''

def cal_cost(G, tree_list, alpha, vnf_type):
    trans_cost = 0
    proc_cost = 0
    for tree in tree_list:
        for (n1,n2,dic) in tree.edges(data=True):
            trans_cost += (dic['data_rate'] / G.edges[n1,n2]['bandwidth'] + 1)
    for (n,dic) in G.nodes(data=True):
        for vnf in dic['vnf']:
            proc_cost += (vnf_type[vnf] / G.nodes[n]['capacity'])
    #print(trans_cost, proc_cost)
    return alpha * trans_cost + (1 - alpha) * proc_cost

'''
count the instance of all VNFs in Graph
'''
def count_instance(G):
    sum = 0;
    for (n, dic) in G.nodes(data=True):
        sum += len(dic['vnf'])
    return sum

'''
find common path
'''
def find_common_path_len(dst, shortest_path_set):
    min_path_length = min(len(shortest_path_set[d]) for d in shortest_path_set)
    move_dst = set()
    common_length = 0
    #union_unfinish_dst_path = []
    # find union path of dst
    for i in range(min_path_length):
        common_flag = 0
        for d in shortest_path_set:
            if d != dst and d not in move_dst and shortest_path_set[dst][i] == shortest_path_set[d][i] and common_flag == 0:
                common_flag = 1
            if shortest_path_set[dst][i] != shortest_path_set[d][i]:
                move_dst.add(d)
            if common_flag == 1:
                common_length += 1
            common_flag = 0
        #union_unfinish_dst_path.append(shortest_path_set[dst_list[0]][i])
    return common_length

def add_new_edge(G, path, path_set, dst, e, data_rate):
    if path.has_edge(*e) == False:
        path.add_edge(*e,data_rate=data_rate)
        G.edges[e]['data_rate'] += data_rate
        G.edges[e]['bandwidth'] -= data_rate
    else:
        commom_len = find_common_path_len(dst, path_set)
        if len(path_set[dst]) > commom_len:
            path.edges[e]['data_rate'] += data_rate
            G.edges[e]['data_rate'] += data_rate
            G.edges[e]['bandwidth'] -= data_rate
