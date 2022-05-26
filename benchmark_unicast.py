import networkx as nx
import copy

import experience_setting as exp_set
import Graph

def add_new_edge(G, path, e, data_rate):
    if path.has_edge(*e) == False:
        path.add_edge(*e,data_rate=data_rate[2],data=[data_rate[1]])
    else:
        path.edges[e]['data_rate'] += data_rate[2]
        path.edges[e]['data'].append(data_rate[1])

    G.edges[e]['data_rate'] += data_rate[2]
    G.edges[e]['bandwidth'] -= data_rate[2]

'''
Use Dijkstra algorithm find the unicast path from src to each dst.
And placing ordered VNF from path head. 
A VNF is dedicated for one dst, not share with other dsts.
'''
def search_unicast_path(G, pos, service, quality_list):
    video_type = [q for q in quality_list]

    src = service[0]
    dsts = service[1]
    sort_dsts = sorted(service[1].items(), key=lambda d: video_type.index(d[1]), reverse=True)

    dst_list = list(d for d in dsts)

    best_quality = service[3]

    sfc = copy.deepcopy(service[2])
    require_quality = set(dsts[i] for i in dsts)
    max_transcoder_num = len(require_quality) - 1
    
    G_min = copy.deepcopy(G)
    unicast_path_min = nx.Graph()
    failed_dsts = list()

    # Record the satisfied vnf now and place node
    index_sfc = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list)
    # Record all path from src to dst with its ordered nodes
    shortest_path_set = {}

    # Reocrd the current quality send to dst
    data_rate = dict()
    for d in dst_list:
        data_rate[d] = [(src, dsts[d], quality_list[dsts[d]])]

    # Find shortest path
    for d in sort_dsts:
        G_tmp = copy.deepcopy(G_min)
        unicast_path_tmp = copy.deepcopy(unicast_path_min)
        dst = d[0]
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, src, dst, weight='weight')
        shortest_path_set[dst] = [src]

        # Update nodes of tmp_unicast_path with nodes of shortest_path
        # node_attr = {node: [placement VNF]}
        node_attr = {}
        for m in shortest_path:
            if m not in unicast_path_min or len(unicast_path_min.nodes[m]) == 0:
                node_attr[m] = {'vnf': []}
            else:
                node_attr[m] = unicast_path_min.nodes[m]

        unicast_path_min.add_nodes_from(shortest_path)
        nx.set_node_attributes(unicast_path_min, node_attr)

        is_enough = True
        for i in range(len(shortest_path)-1):
            shortest_path_set[dst].append(shortest_path[i+1])

            # Don't place vnf on src and dst
            if shortest_path[i] != src and shortest_path[i] != dst:
                j = shortest_path[i]

                if index_sfc[dst]['index'] >= len(sfc[dst])-1:
                    #output_q = sort_quality[index_sfc[dst]['index']+1]
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

                if data_rate[dst][-1][0] != j:
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
            
            e = (shortest_path[i],shortest_path[i+1])
            if G_min.edges[e]['bandwidth'] < data_rate[dst][-1][2]:
                is_enough = False
                continue
            add_new_edge(G_min, unicast_path_min, e, data_rate[dst][-1])

        if is_enough == False:
            G_min = copy.deepcopy(G_tmp)
            failed_dsts.append(dst)
            unicast_path_min = copy.deepcopy(unicast_path_tmp)

    for d in sfc:
        if d in failed_dsts:
            sfc[d] = []
            shortest_path_set[d] = []
            data_rate[d] = []

    # print('============')
    # print(dsts)
    # print("shortest_path_set",shortest_path_set)
    # print("date_rate ",data_rate)
    # print("fail ", failed_dsts)
    # print('============')

    weight = (0.6, 0.4, 1)
    # print(Graph.cal_total_cost(G_min, weight, False))

    return (G_min, unicast_path_min, failed_dsts, sfc, shortest_path_set)

