import matplotlib.pyplot as plt
import networkx as nx
from numpy import true_divide

'''
Calculate the cost of multicast path

total cost = transmission cost + processing cost + placing cost
'''
def cal_total_cost(G):
    trans_cost = 0
    proc_cost = 0
    plac_cost = 0

    for (n1,n2,dic) in G.edges(data=True):
        trans_cost += dic['data_rate']
    for (n,dic) in G.nodes(data=True):
        if len(dic['vnf']) != 0:
            plac_cost += len(set(v[0] for v in dic['vnf']))
            for vnf in dic['vnf']:
                proc_cost += vnf[2]

    return trans_cost + proc_cost + plac_cost

'''
print graph
'''
def printGraph(G, pos, filename, service, weight):
    color_node = []
    dst_list = list(d for d in service[1])
    for n in G:
        if n == service[0]:
            color_node.append('#E3432B')
        elif n in dst_list:
            color_node.append('#FFD448')  
        else:
            color_node.append('gray')

    edge_labels = dict([((n1, n2), round(d[weight],3)) for n1, n2, d in G.edges(data=True)])
    
    plt.figure(figsize=(8,8))
    if filename in ['tmp', 'G_min_miss_vnf', 'path_miss_vnf', 'G_min_remove', 'path_remove', 'G_min_update', 'path_update', 'G_final_0', 'path_final_0', 'G_final_1', 'path_final_1','G_final_2', 'path_final_2', 'path_final']: 
        node_labels = dict([(n, d['vnf']) for (n,d) in G.nodes(data=True)])
        pos_nodes = dict((node,(coords[0], coords[1]-0.05)) for node, coords in pos.items())
        nx.draw_networkx_labels(G, pos=pos_nodes, labels=node_labels, font_color='blue', font_size=8)
        color_edge = []
        for n1,n2,d in G.edges(data=True):
            if d['data_rate'] != 0:
                color_edge.append('g')
            else:
                color_edge.append('k') 
        nx.draw(G, pos, node_color=color_node, edge_color=color_edge, with_labels=True, font_size=6, node_size=200)
    else:
        nx.draw(G, pos, node_color=color_node, with_labels=True, font_size=6, node_size=200)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.savefig('img/'+filename+'.png')
    plt.close()

'''
Count the instance of all VNFs in Graph
'''
def count_instance(G):
    sum = 0;
    for (n, dic) in G.nodes(data=True):
        sum += len(dic['vnf'])
    return sum

'''
Find common path for adding new edge
'''
def find_common_path_len_edge(path, dst, shortest_path_set, data_rate, video_type):
    move_dst = [dst]
    common_length = 0
    
    # Find union path of dst
    for i in range(len(shortest_path_set[dst])-1):
        common_flag = 0
        for d in shortest_path_set:
            if len(shortest_path_set[d])-1 <= i:
                move_dst.append(d)
            # Check if there have better quality been through the edge
            if d not in move_dst and shortest_path_set[dst][i] == shortest_path_set[d][i] and common_flag == 0:
                e = (shortest_path_set[d][i], shortest_path_set[d][i+1])
                index = video_type.index(data_rate[1])
                edge_video_list = [video_type.index(q) for q in path.edges[e]['data']]
                
                if index <= max(edge_video_list):
                    common_flag = 1
            if d not in move_dst and shortest_path_set[dst][i] != shortest_path_set[d][i]:
                move_dst.append(d)
            if common_flag == 1:
                common_length += 1
            common_flag = 0
    return common_length

'''
Find common path for adding new vnf
'''
def find_common_path_len_node(path, dst, shortest_path_set, data_rate):
    move_dst = [dst]
    common_length = 1
    
    # Find union path of dst
    for i in range(len(shortest_path_set[dst])):
        common_flag = 0
        for d in shortest_path_set:
            if len(shortest_path_set[d]) <= i:
                move_dst.append(d)
            # Check if there have better quality been through the edge
            if d not in move_dst and shortest_path_set[dst][i] == shortest_path_set[d][i] \
                and common_flag == 0 and data_rate[1] in list(q[1] for q in path.nodes[shortest_path_set[d][i]]['vnf']):
                common_flag = 1
                
            if d not in move_dst and shortest_path_set[dst][i] != shortest_path_set[d][i]:
                move_dst.append(d)
            if common_flag == 1:
                common_length += 1
            common_flag = 0

    return common_length

def add_new_edge(G, path, path_set, dst, e, data_rate, video_type):
    #print('add_new_edge')
    if path.has_edge(*e) == False:
        path.add_edge(*e,data_rate=data_rate[2],data=[data_rate[1]])
        G.edges[e]['data_rate'] += data_rate[2]
        G.edges[e]['bandwidth'] -= data_rate[2]
    else:
        commom_len = find_common_path_len_edge(path, dst, path_set, data_rate, video_type)
        if len(path_set[dst])-1 > commom_len:
            path.edges[e]['data_rate'] += data_rate[2]
            path.edges[e]['data'].append(data_rate[1])
            G.edges[e]['data_rate'] += data_rate[2]
            G.edges[e]['bandwidth'] -= data_rate[2]

def add_new_vnf(G, path, path_set, dst, node, vnf, data_rate):
    commom_len = find_common_path_len_node(path, dst, path_set, data_rate)
    if path_set[dst].index(node) >= commom_len:
        G.nodes[node]['vnf'].append((vnf, data_rate[1],data_rate[2]))
        path.nodes[node]['vnf'].append((vnf, data_rate[1], data_rate[2]))
        G.nodes[node]['capacity'] -= data_rate[2]
        return True
    return False
    