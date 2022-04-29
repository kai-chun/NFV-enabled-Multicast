import matplotlib.pyplot as plt
import networkx as nx

'''
Calculate the cost of multicast path

total cost = transmission cost + processing cost + placing cost
'''
# weight = (transmission weight, processing weight, placing weight)
def cal_total_cost(G, weight):
    trans_cost = 0
    proc_cost = 0
    plac_cost = 0
    total_cost = 0

    for (n1,n2,dic) in G.edges(data=True):
        trans_cost += dic['data_rate']
    for (n,dic) in G.nodes(data=True):
        if len(dic['vnf']) != 0:
            plac_cost += len(set(v[0][0] for v in dic['vnf']))
            for vnf in dic['vnf']:
                proc_cost += vnf[2]
    total_cost = weight[0]*trans_cost + weight[1]*proc_cost +weight[2]*plac_cost
    return (total_cost, trans_cost, plac_cost, proc_cost)

def cal_trans_cost(G):
    trans_cost = 0
    for (n1,n2,dic) in G.edges(data=True):
        trans_cost += dic['data_rate']
    return trans_cost

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
def find_common_path_len_edge(dst, shortest_path_set, all_data_rate):
    move_dst = [dst]
    common_length = 1
    
    # Find union path of dst
    for i in range(1,len(shortest_path_set[dst])):
        common_flag = 0
        for d in shortest_path_set:
            if len(shortest_path_set[d])-1 < i:
                move_dst.append(d)
            # Check if there have better quality been through the edge
            if d not in move_dst and shortest_path_set[dst][i] == shortest_path_set[d][i]:
                if all_data_rate[dst][i-1][1] == all_data_rate[d][i-1][1]:
                    # print(shortest_path_set[dst], shortest_path_set[d], end=' ')
                    common_flag = 1
                else:
                    move_dst.append(d)
            if d not in move_dst and shortest_path_set[dst][i] != shortest_path_set[d][i]:
                move_dst.append(d)
        if common_flag == 1:
            common_length += 1

    return common_length

'''
Find common path for adding new vnf processing data
'''
def find_common_path_len_node(path, dst, vnf, shortest_path_set, all_data_rate):
    move_dst = [dst]
    common_length = 1
    
    # Find union path of dst
    for i in range(1, len(shortest_path_set[dst])):
        common_flag = 0
        for d in shortest_path_set:
            if len(shortest_path_set[d])-1 <= i:
                move_dst.append(d)

            # Check if there have better quality been through the edge
            if d not in move_dst and shortest_path_set[dst][i] == shortest_path_set[d][i] \
                and vnf in list(q[0] for q in path.nodes[shortest_path_set[dst][i]]['vnf']):
                # Make sure that the processing data are same (input and output data)
                if len(all_data_rate[dst]) > i and all_data_rate[dst][i-1][1] == all_data_rate[d][i-1][1] and all_data_rate[dst][i][1] == all_data_rate[d][i][1]:
                    common_flag = 1
                else:
                    move_dst.append(d)
            if d not in move_dst and shortest_path_set[dst][i] != shortest_path_set[d][i]:
                move_dst.append(d)
        if common_flag == 1:
            common_length += 1
    return common_length

def find_common_path_len_node_main(dsts, shortest_path_set, all_data_rate):
    move_dst = dsts
    common_length = 1
    dst = dsts[0]
    
    # Find union path of dst
    for i in range(1, len(shortest_path_set[dst])):
        common_flag = 0
        for d in shortest_path_set:
            if len(shortest_path_set[d])-1 < i:
                move_dst.append(d)
            
            # Check if there have better quality been through the edge
            if d not in move_dst and shortest_path_set[dst][i] == shortest_path_set[d][i]:
                # Make sure that the processing data are same (input and output data)
                if len(all_data_rate[dst]) > i and len(all_data_rate[d]) > i and all_data_rate[dst][i-1][1] == all_data_rate[d][i-1][1] and all_data_rate[dst][i][1] == all_data_rate[d][i][1]:
                    common_flag = 1
                else:
                    move_dst.append(d)
            if d not in move_dst and shortest_path_set[dst][i] != shortest_path_set[d][i]:
                move_dst.append(d)
        if common_flag == 1:
            common_length += 1
    return common_length

def add_new_edge(G, path, path_set, dst, e, data_rate, all_data_rate):  
    if path.has_edge(*e) == False:
        path.add_edge(*e,data_rate=data_rate[2],data=[data_rate[1]])
        G.edges[e]['data_rate'] += data_rate[2]
        G.edges[e]['bandwidth'] -= data_rate[2]
    else:
        commom_len = find_common_path_len_edge(dst, path_set, all_data_rate) 
        if len(path_set[dst]) > commom_len:
            path.edges[e]['data_rate'] += data_rate[2]
            path.edges[e]['data'].append(data_rate[1])
            G.edges[e]['data_rate'] += data_rate[2]
            G.edges[e]['bandwidth'] -= data_rate[2]
    
def add_new_processing_data(G, path, path_set, dst, node, vnf, data_rate, all_data_rate):
    for n in path.nodes:
        if 'vnf' not in path.nodes[n]:
            path.add_node(n, vnf=[])
    commom_len = find_common_path_len_node(path, dst, vnf, path_set, all_data_rate)

    ## index 若有重複時怎麼確定現在要看的是哪個node
    index = path_set[dst].index(node)
    if path_set[dst].count(node) > 1:
        for i in range(len(path_set[dst])-1,-1,-1):
            if path_set[dst][i] == node:
                index = i
                break

    if index >= commom_len:
        if G.nodes[node]['com_capacity'] >= data_rate[2]: # check compute capacity
            G.nodes[node]['vnf'].append((vnf, data_rate[1],data_rate[2]))
            path.nodes[node]['vnf'].append((vnf, data_rate[1], data_rate[2]))
            G.nodes[node]['com_capacity'] -= data_rate[2]
            return True
        else:
            return False
    return True

def add_new_processing_data_main(G, path, path_set, dst, index, vnf, data_rate, all_data_rate):
    for n in path_set[dst]:
        if n not in path.nodes:
            path.add_node(n, vnf=[])
    commom_len = find_common_path_len_node_main([dst], path_set, all_data_rate)
    
    node = path_set[dst][index]

    if index >= commom_len:
        if G.nodes[node]['com_capacity'] >= data_rate[2]: # check compute capacity
            G.nodes[node]['vnf'].append((vnf, data_rate[1],data_rate[2]))
            path.nodes[node]['vnf'].append((vnf, data_rate[1], data_rate[2]))
            G.nodes[node]['com_capacity'] -= data_rate[2]
            G.nodes[node]['share_num'] += 1
            return True
        else:
            return False
    
    G.nodes[node]['share_num'] += 1
    return True