import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import re
import math

# class Topology:
#     def __init__(self, node_num, edge_prob):
#         self.node_num = node_num  # number of node in topology

# class Exp:
#     def __init__(self, topology, node_num, dst_ratio):
        
#         self.dst_ratio = dst_ratio  # percentage of dst in node
#         self.dst_num = int(ceil(node_num * dst_ratio))

def create_topology(node_num, edge_prob):
    ### Use realistic topology
    # input_G = nx.read_gml("topology/Abilene.gml")

    # pos = {}
    # node_list = {}
    # for i,n in enumerate(input_G.nodes(data=True)):
    #     node_list[n[0]] = i
    #     pos[i] = [n[1]['Longitude'],n[1]['Latitude']]

    # edge_list = []
    # edge_attr = {}
    # for n1, n2, dic in input_G.edges(data=True):
    #     e = (node_list[n1], node_list[n2])
    #     edge_list.append(e)
    #     link_label = dic['LinkLabel'].split(' ')
    #     edge_attr[e] = {}
    #     #edge_attr[e]['bandwidth'] = int(link_label[0])
    #     #edge_attr[e]['data_rate'] = 0

    ### Grid Grpah
    m = int(math.floor(math.sqrt(node_num)))
    input_G = nx.generators.grid_2d_graph(m,m)
    
    node_list = {}
    for i,n in enumerate(input_G.nodes(data=True)):
        node_list[n[0]] = i

    edge_list = []
    for n1, n2, dic in input_G.edges(data=True):
        e = (node_list[n1], node_list[n2])
        edge_list.append(e)

    G = nx.Graph()
    G.add_nodes_from(list(range(len(node_list))))
    G.add_edges_from(edge_list)
    pos = nx.spectral_layout(G)
    
    # nx.set_node_attributes(G, {n: {'capacity': round(random.uniform(2.5,10), 2)} for n in G.nodes})
    # nx.set_node_attributes(G, {n: {'vnf': []} for n in G.nodes})
    # nx.set_edge_attributes(G, edge_attr)

    ### Random
    # random_seed = 138
    # G = nx.gnp_random_graph(node_num, edge_prob, random_seed)
    # while nx.is_connected(G) == False:
    #     G = nx.gnp_random_graph(node_num, edge_prob, random_seed)
    # pos = nx.spring_layout(G) 

    nx.set_node_attributes(G, {n: {'capacity': round(random.uniform(2.5,10), 2)} for n in G.nodes})
    nx.set_node_attributes(G, {n: {'vnf': []} for n in G.nodes})
    nx.set_edge_attributes(G, {e: {'bandwidth': round(random.uniform(2.5,10), 2)} for e in G.edges})
    nx.set_edge_attributes(G, {e: {'data_rate': 0} for e in G.edges})

    nx.draw(G, pos)
    plt.savefig('img/Grid.png')
    plt.close()

    #print(G.nodes(data=True))
    #print(G.edges(data=True))

    return (G, pos)

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
    bitrate_table = {'360p': {'360p': 1} \
                    ,'480p': {'360p': 0.7, '480p': 1} \
                    ,'720p': {'360p': 0.5, '480p': 0.7, '720p': 1} \
                    ,'1080p': {'360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}}
    return round(bitrate_table[input_q][output_q] * data_rate, 2)

'''
Generate the experience data
'''
def generate_exp(graph_exp, service_exp, order, factor, factor_num):
    # Generate topology
    tmp = create_topology(graph_exp['node_num'], graph_exp['edge_prob'])
    G = tmp[0]
    pos = tmp[1]
    N = G.nodes()
      
    output_file = open('exp_setting/'+str(factor)+'/G_'+str(factor_num)+'_'+str(order)+'.txt', 'w')
    for n,dic in G.nodes(data=True):
        output_file.write(str(n)+" "+str(dic['capacity'])+"\n")

    output_file.write("\n")

    for n1,n2,dic in G.edges(data=True):
        output_file.write(str(n1)+" "+str(n2)+" "+str(dic['bandwidth'])+"\n")
    
    output_file.write("\n")

    for n in pos:
        output_file.write(str(n)+" "+str(pos[n][0])+" "+str(pos[n][1])+"\n")
    
    output_file.close()
 
    # Generate service
    src = random.choice(list(N))
    dsts = create_client(service_exp['dst_num'], N, src, service_exp['video_type'])
    #dst_list = list(d for d in dsts)
    # sfc = ['t']
    # service = (src, dsts, sfc, quality_list['1080p'])
    
    output_file_client = open('exp_setting/'+str(factor)+'/service_'+str(factor_num)+'_'+str(order)+'.txt', 'w')
    output_file_client.write(str(src)+"\n")
    for d in dsts:
        output_file_client.write(str(d)+" "+str(dsts[d])+"\n")
    output_file_client.close()

'''
Read the experience data of graph
'''
def read_exp_graph(order, factor, factor_num):
    input_file = open('exp_setting/'+str(factor)+'/G_'+str(factor_num)+'_'+str(order)+'.txt', 'r', 1)
    content = input_file.readlines()

    index = 0
    node_list = []
    for i in range(len(content)):
        if content[i] == "\n": 
            index = i+1
            break
        line = list(re.split(' ', content[i]))
        node = (int(line[0]), {'capacity': float(line[1][:-1]), 'vnf': []})
        node_list.append(node)

    edge_list = []
    for i in range(index, len(content)):
        if content[i] == "\n": 
            index = i+1
            break
        line = list(re.split(' ', content[i]))
        edge = (int(line[0]), int(line[1]), {'bandwidth': float(line[2][:-1]), 'data_rate': 0})
        edge_list.append(edge)

    pos = {}
    for i in range(index, len(content)):
        line = list(re.split(' ', content[i]))
        pos[int(line[0])] = [float(line[1]),float(line[2])]

    input_file.close()

    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)
    
    return (G, pos)

'''
Read the experience data of service
'''
def read_exp_service(order, factor, factor_num):
    input_data = open('exp_setting/'+str(factor)+'/service_'+str(factor_num)+'_'+str(order)+'.txt', 'r', 1)
    src = int(input_data.readline())

    content = input_data.readlines()
    dsts = {}
    for c in content:
        line = list(re.split(' ', c))
        dsts[int(line[0])] = line[1][:-1]
    
    input_data.close()
    
    return (src, dsts)
    