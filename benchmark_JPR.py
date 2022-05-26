'''
Find the multicast path with the minimum cost.

input: 
G' = (N',E') = the copy of G, N = nodes(include M = NFV nodes, F = switches), L = links
S = (s,D,V) = request(service), s = source, D = set of dsts, V = {f1,f2,...} = SFC

output:
Gv = multicast path
'''

import copy
import sys
import networkx as nx

import Graph
import experience_setting as exp_set

'''
Calculate the weight of G.link
'''
def cal_link_weight(G, edge, alpha, beta, data_rate):
    if G.edges[edge]['bandwidth'] == 0 and G.nodes[edge[1]]['com_capacity'] == 0:
        weight = alpha * (1 + 1) + beta * (1)
    elif G.edges[edge]['bandwidth'] == 0: 
        weight = alpha * (1 + 1) + beta * (data_rate / G.nodes[edge[1]]['com_capacity'])
    elif G.nodes[edge[1]]['com_capacity'] == 0:
        weight = alpha * (data_rate / G.edges[edge]['bandwidth'] + 1) + beta * (1)
    else:
        weight = alpha * (data_rate / G.edges[edge]['bandwidth'] + 1) + beta * (data_rate / G.nodes[edge[1]]['com_capacity'])
    return weight

'''
Calculate the cost of multicast path in benchmark method

total cost = transmission cost + processing cost
'''
def cal_cost(G, tree_list, alpha, vnf_type):
    trans_cost = 0
    proc_cost = 0
    for tree in tree_list:
        for (n1,n2,dic) in tree.edges(data=True):
            bandwidth = G.edges[n1,n2]['bandwidth']
            if bandwidth == 0:
                bandwidth = 1e-8
            trans_cost += (dic['data_rate'] / bandwidth + 1)
        for (n,dic) in tree.nodes(data=True):
            if len(dic['vnf']) == 0:
                proc_cost += 0
            else:
                for proc_data in dic['vnf']:
                    capacity = G.nodes[n]['com_capacity']
                    if capacity == 0:
                        capacity = 1e-8
                    proc_cost += (proc_data[2] / capacity)
    return alpha * trans_cost + (1 - alpha) * proc_cost

'''
Remove the wrong path which unsatisfy the SFC.
Also update the right data_rate and path.
'''
def update_path(sfc, index_sfc, sort_dsts, G, G_min, multicast_path, shortest_path_set, data_rate):
    missing_vnf_dsts = []
    for dst in index_sfc: # find all dst which missing vnf 
        if index_sfc[dst]['index'] < len(sfc[dst])-1:
            missing_vnf_dsts.append(dst)

    tmp_d = list(data_rate.keys())[0]
    src = data_rate[tmp_d][0][0]
    best_quality = data_rate[tmp_d][0][1]
    data_size = data_rate[tmp_d][0][2]

    if len(missing_vnf_dsts) > 0:
        # Remove unnecessary edges: the path from last vnf node to dst
        update_shortest_path_set = copy.deepcopy(shortest_path_set)
        for dst in missing_vnf_dsts:
            last_place_node = index_sfc[dst]['place_node'][-1] if index_sfc[dst]['index'] > -1 else -1
            last_node_index = shortest_path_set[dst].index(last_place_node) if last_place_node != -1 else 0
            delete_node = update_shortest_path_set[dst][last_node_index+1:]
            for i in delete_node:
                if i in multicast_path.nodes() and multicast_path.nodes[i]['vnf'] != []:
                    delete_node.remove(i)
            multicast_path.remove_nodes_from(delete_node)
            if dst not in multicast_path.nodes():
                multicast_path.add_node(dst, vnf=[])
            del update_shortest_path_set[dst][last_node_index+1:]
            
        for dst in update_shortest_path_set:
            for i in range(len(data_rate[dst])-1,-1,-1):
                data = data_rate[dst][i]
                if data[0] not in update_shortest_path_set[dst]:
                    data_rate[dst].pop(i)
                if len(update_shortest_path_set[dst]) == 1:
                    data_rate[dst] = [(src, best_quality, data_size)]
                    break

        # Rebuilding the multicast path with update_shortest_path_set
        G_min.remove_edges_from(G_min.edges())
        for n1,n2,d in G.edges(data=True):
            G_min.add_edge(n1,n2,data_rate=d['data_rate'],bandwidth=d['bandwidth'])
        
        multicast_path.remove_edges_from(multicast_path.edges())
        tmp_list = dict()
        for d in sort_dsts:
            dst = d[0]
            if dst not in index_sfc: continue
            tmp_list[dst] = [update_shortest_path_set[dst][0]]
            for i in range(1,len(update_shortest_path_set[dst])):
                e = (update_shortest_path_set[dst][i-1], update_shortest_path_set[dst][i])
    
                tmp_list[dst].append(update_shortest_path_set[dst][i])
                Graph.add_new_edge(G_min, multicast_path, tmp_list, dst, e, data_rate[dst][i-1], data_rate)
    else:
        update_shortest_path_set = copy.deepcopy(shortest_path_set)
        tmp_list = dict()
        #print('no miss')
    return (missing_vnf_dsts, update_shortest_path_set, tmp_list)

# service = (src, dsts, sfc, data_rate)
# alpha: the weight of link transmission cost
# quality_list: all kinds of video quality
def search_multipath(G, service, alpha, vnf_type, quality_list, isReuse):
    N = G.nodes()

    video_type = [q for q in quality_list]

    src = service[0]
    dsts = service[1]
    sort_dsts = sorted(service[1].items(), key=lambda d: video_type.index(d[1]), reverse=True)

    beta = 1 - alpha # the weight of processing cost
    dst_list = list(d for d in dsts)

    best_quality = service[3]

    sfc = copy.deepcopy(service[2])
    require_quality = set(dsts[i] for i in dsts)
    max_transcoder_num = len(require_quality) - 1
    sort_quality = sorted(require_quality, key=lambda q: video_type.index(q), reverse=True)

    # If dsts need different quality of video then place a transcoder
    if max_transcoder_num > 0: 
        for d in dst_list:
            if dsts[d] != best_quality:
                if isReuse:
                    sfc[d].append('t')
                else:
                    sfc[d].append('t_'+str(d))

    # print('--- environment ---')
    # print('src = ', src, ", dst = ", dsts)
    # print(sort_dsts)
    # print("sfc = ", sfc)
    # print('===============')

    G_min = copy.deepcopy(G) # record the min cost network 
    multicast_path_min = nx.Graph() # record the min cost path
    min_data_rate = dict() # record the min cost data_rate

    # Record the satisfied vnf of min cost path and place node
    # index_sfc_min = {dst: {'index': the index of satisdied VNF,'place_node': [the nodes which satisdied VNFs placing]}}
    index_sfc_min = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list) 

    for node in N:
        # copy G to G_new, and update the weight of edges
        G_new = copy.deepcopy(G)
        nx.set_edge_attributes(G_new, {e: {'weight': cal_link_weight(G_new,e,alpha,beta,quality_list[best_quality])} for e in G_new.edges})

        # Generate metric closure
        G_metric_closure = nx.algorithms.approximation.metric_closure(G_new,weight='weight')

        # Generate the set of {src, dsts, n} in G_new
        G_keyNode = nx.Graph()
        G_keyNode.add_nodes_from([src,node])
        G_keyNode.add_nodes_from(dst_list)

        for n1,n2,d in G_metric_closure.edges(data=True):
            if n1 in G_keyNode and n2 in G_keyNode:
                G_keyNode.add_edge(n1,n2,weight=d['distance'])

        # Generate MST
        T = nx.minimum_spanning_tree(G_keyNode,weight='distance')

        # Replace edges in MST with corresponding paths in G_new
        keynode_path = nx.Graph(T)
        for e in T.edges:
            if e not in G:
                keynode_path.remove_edge(e[0],e[1])
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_new, e[0],e[1], weight='weight')
                for j in shortest_path:
                    for i in range(len(shortest_path)-1):
                        keynode_path.add_edge(shortest_path[i],shortest_path[i+1],data_rate=0)

        tmp_G = copy.deepcopy(G) # record the temporary capacity and vnf of nodes
        tmp_multicast_path = nx.Graph() # record the temporary multicast path

        # Record the satisfied vnf now and place node
        # index_sfc = {dst: {'index': the index of satisdied VNF,'place_node': [the nodes which satisdied VNFs placing]}}
        index_sfc = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list)
        
        # Record all path from src to dst with its ordered nodes
        # shortest_path_set = {dst: [the ordered nodes of shortest path]}
        shortest_path_set = {}

        # Reocrd the current quality send to dst
        # data_rate = {dst: [(node of edge, video quality, data_rate of quality)]}
        data_rate = dict()
        for d in dst_list:
            data_rate[d] = [(src, best_quality, quality_list[best_quality])]

        '''
        Greedy placement VNF on nodes (for each dst)
        that its capacity can satisfy the data_rate of VNF
        '''
        for d in sort_dsts:
            dst = d[0]
            shortest_path = nx.algorithms.shortest_paths.dijkstra_path(keynode_path, src, dst, weight='weight')
            shortest_path_set[dst] = [src]
            
            # Update nodes of tmp_multicast_path with nodes of shortest_path
            # node_attr = {node: [placement VNF]}
            node_attr = {}
            for m in shortest_path:
                if m not in tmp_multicast_path:
                    node_attr[m] = {'vnf': []}
                else:
                    node_attr[m] = tmp_multicast_path.nodes[m]
            tmp_multicast_path.add_nodes_from(shortest_path)
            nx.set_node_attributes(tmp_multicast_path, node_attr)

            # Iterative the node in shortes_path from src to dst
            for i in range(len(shortest_path)-1):
                shortest_path_set[dst].append(shortest_path[i+1])

                # Don't place vnf on src and dst
                if shortest_path[i] != src and shortest_path[i] != dst:
                    j = shortest_path[i]

                    if index_sfc[dst]['index'] >= len(sfc[dst])-1:
                        data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                    
                    # Find node to place vnf until finishing sfc
                    while index_sfc[dst]['index'] < len(sfc[dst])-1:
                        vnf = sfc[dst][index_sfc[dst]['index']+1]

                        # There are vnf instance on node j
                        if isReuse and vnf in list(v[0] for v in tmp_G.nodes[j]['vnf']):
                            # Processing data with transcoder
                            if "t" in vnf: # This vnf is transcoder
                                output_q = dsts[dst]
                                update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                                data_rate[dst].append((j,output_q,update_data_rate))
                            else:
                                data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            
                            is_process_data = Graph.add_new_processing_data(tmp_G, tmp_multicast_path, shortest_path_set, dst, j, vnf, data_rate[dst][-2], data_rate)
                            
                            if is_process_data == True:
                                index_sfc[dst]['index'] += 1
                                index_sfc[dst]['place_node'].append(j)
                            #else:
                                #data_rate[dst].pop(-1)
                        
                        if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                            break
                        
                        # Check if the node has enough resource to place instance
                        if tmp_G.nodes[j]['mem_capacity'] > 0: # memory capacity
                            # add new vnf instance
                            if vnf not in list(v[0] for v in tmp_multicast_path.nodes[j]['vnf']): 
                                tmp_G.nodes[j]['mem_capacity'] -= 1

                            # Processing data with transcoder
                            if "t" in vnf: # This vnf is transcoder
                                output_q = dsts[dst]
                                update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                                data_rate[dst].append((j,output_q,update_data_rate))
                            else:
                                data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            
                            is_process_data = Graph.add_new_processing_data(tmp_G, tmp_multicast_path, shortest_path_set, dst, j, vnf, data_rate[dst][-2], data_rate)
                            
                            if is_process_data == True:
                                index_sfc[dst]['index'] += 1
                                index_sfc[dst]['place_node'].append(j)
                            else:
                                #data_rate[dst].pop(-1)
                                if vnf not in list(v[0] for v in tmp_multicast_path.nodes[j]['vnf']):
                                    tmp_G.nodes[j]['mem_capacity'] += 1
                                break
                        else:
                            break
                    if data_rate[dst][-1][0] != j:
                        data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

                # Update the bandwidth of link
                e = (shortest_path[i],shortest_path[i+1])
                Graph.add_new_edge(tmp_G, tmp_multicast_path, shortest_path_set, dst, e, data_rate[dst][-1], data_rate)

            #Graph.printGraph(tmp_multicast_path, pos, 'tmp', service, 'data_rate')
        
        # Greedy selection
        if nx.classes.function.is_empty(multicast_path_min):
            multicast_path_min = copy.deepcopy(tmp_multicast_path)
            G_min = copy.deepcopy(tmp_G)
            index_sfc_min = copy.deepcopy(index_sfc)
            min_data_rate = copy.deepcopy(data_rate)
        elif Graph.count_instance(tmp_multicast_path) == Graph.count_instance(multicast_path_min) and \
            cal_cost(tmp_G, [tmp_multicast_path], alpha, vnf_type) < cal_cost(G_min, [multicast_path_min], alpha, vnf_type):
                multicast_path_min = copy.deepcopy(tmp_multicast_path)
                G_min = copy.deepcopy(tmp_G)
                index_sfc_min = copy.deepcopy(index_sfc)
                min_data_rate = copy.deepcopy(data_rate)
        elif Graph.count_instance(tmp_multicast_path) > Graph.count_instance(multicast_path_min):
                multicast_path_min = copy.deepcopy(tmp_multicast_path)
                G_min = copy.deepcopy(tmp_G)
                index_sfc_min = copy.deepcopy(index_sfc)
                min_data_rate = copy.deepcopy(data_rate)

    # Graph.printGraph(G_min, pos, 'G_min_miss_vnf', service, 'bandwidth')
    # Graph.printGraph(multicast_path_min, pos, 'path_miss_vnf', service, 'data_rate')

    # If dsts have same quality of video then return the path without VNF
    # if max_transcoder_num == 0:
    #     print('no vnf')
    #print('greedy ok')
    
    '''
    A corrective subroutine that places the missing NF instances 
    on the closest NFV node from the multicast topology.
    '''
    tmp = update_path(sfc, index_sfc_min, sort_dsts, G, G_min, multicast_path_min, shortest_path_set, min_data_rate)
    missing_vnf_dsts = tmp[0]
    update_shortest_path_set = tmp[1]

    # Graph.printGraph(G, pos, 'G', service, 'bandwidth')
    # Graph.printGraph(G_min, pos, 'G_min_remove', service, 'bandwidth')
    # Graph.printGraph(multicast_path_min, pos, 'path_remove', service, 'data_rate')
    
    # Fill up the empty min_data_rate[dst] with best quality (initial date to send)
    for d in dst_list:
        if len(min_data_rate[d]) == 0:
            min_data_rate[d] = [(src, best_quality, quality_list[best_quality])]

    # Place vnf nearest the common path
    for d in sort_dsts:
        dst = d[0]
        if dst not in missing_vnf_dsts:
            continue

        find_distance = 0 # the length range of finding node away from last_node that placing VNF
        place_flag = 0 # whether placing VNF or not (share or initiate), 0: not place, 1: place, 2: the node can't place
        
        while index_sfc_min[dst]['index'] < len(sfc[dst])-1:
            last_node = update_shortest_path_set[dst][-1]
            if place_flag != 1:
                find_distance += 1
                if find_distance >= len(G.edges()):
                    print('cannot find path')
                    return (G, nx.Graph(), [], {})
            else:
                find_distance = 1
                place_flag = 0
            
            alternate_nodes_len = nx.single_source_shortest_path_length(G_min, last_node, find_distance)
            alternate_nodes = list(n for n in alternate_nodes_len)

            # Find the node with distance(<= find_distance) between last_node to place the unsatisfied VNF.
            # If find the node to place, then restart to find another node to place next VNF.
            for i in alternate_nodes:
                # if place_flag != 0:
                #     break
                ### 改過 not sure
                if place_flag == 1:
                    break
                
                # If find_distance > 1, the node that distance = 1 have been searched
                # it don't have to consider the node again
                if find_distance > 1 and alternate_nodes_len[i] <= find_distance-1:
                    continue

                last_node = update_shortest_path_set[dst][-1]
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, i, weight='weight')
                
                vnf = sfc[dst][index_sfc_min[dst]['index']+1]

                # Don't place on last_node, src, and dst
                if i == last_node or i == src or i == dst:
                    continue

                # There are vnf instance on node i
                while vnf in list(v[0] for v in G_min.nodes[i]['vnf']):
                    if isReuse == False:
                        break
                    tmp_G = copy.deepcopy(G_min)
                    tmp_path = copy.deepcopy(multicast_path_min)
                    tmp_path_set = copy.deepcopy(update_shortest_path_set)
                    tmp_data_rate = copy.deepcopy(min_data_rate)
                    
                    # Processing data with transcoder
                    if "t" in vnf: # This vnf is transcoder
                        output_q = dsts[dst]
                        update_data_rate = exp_set.cal_transcode_bitrate(min_data_rate[dst][-1][2], min_data_rate[dst][-1][1], output_q)
                        min_data_rate[dst].append((i,output_q,update_data_rate))
                    else:
                        min_data_rate[dst].append((i,min_data_rate[dst][-1][1],min_data_rate[dst][-1][2])) 

                    # If it has placed vnf, there don't need to add edge again.
                    if place_flag != 1:
                        # Add edge to connect node
                        if find_distance == 1:
                            if update_shortest_path_set[dst][-1] != i:
                                update_shortest_path_set[dst].append(i)
                            if i not in multicast_path_min:
                                multicast_path_min.add_node(i, vnf=[])
                            
                            e = (last_node, i)
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, min_data_rate[dst][-2], min_data_rate)
                        else:
                            for j in range(len(shortest_path)-1):
                                if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                    update_shortest_path_set[dst].append(shortest_path[j+1])
                                if shortest_path[j] not in multicast_path_min:
                                    multicast_path_min.add_node(shortest_path[j], vnf=[])
                                e = (shortest_path[j], shortest_path[j+1])
                                if j != 0:
                                    min_data_rate[dst].insert(-2, (shortest_path[j],min_data_rate[dst][-1][1],min_data_rate[dst][-1][2]))
                                    #min_data_rate[dst].append((shortest_path[j],min_data_rate[dst][-1][1],min_data_rate[dst][-1][2]))
                                Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, min_data_rate[dst][-2], min_data_rate)
                    
                    is_process_data = Graph.add_new_processing_data(G_min, multicast_path_min, update_shortest_path_set, dst, i, vnf, min_data_rate[dst][-2], min_data_rate)

                    if is_process_data == True:
                        index_sfc_min[dst]['index'] += 1
                        index_sfc_min[dst]['place_node'].append(i)
                        place_flag = 1
                    else:
                        G_min = copy.deepcopy(tmp_G)
                        multicast_path_min = copy.deepcopy(tmp_path)
                        update_shortest_path_set = copy.deepcopy(tmp_path_set)
                        min_data_rate = copy.deepcopy(tmp_data_rate)
                        place_flag = 2
                        break

                    if index_sfc_min[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                        break
                    vnf = sfc[dst][index_sfc_min[dst]['index']+1]

                if index_sfc_min[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                    break

                # Check if node has enough resource to place instance
                while G_min.nodes[i]['mem_capacity'] >= 0: # memory capacity
                    if i not in multicast_path_min:
                        multicast_path_min.add_node(i, vnf=[])
                    
                    tmp_G = copy.deepcopy(G_min)
                    tmp_path = copy.deepcopy(multicast_path_min)
                    tmp_path_set = copy.deepcopy(update_shortest_path_set)
                    tmp_data_rate = copy.deepcopy(min_data_rate)

                    for n in multicast_path_min.nodes:
                        if 'vnf' not in multicast_path_min.nodes[n]:
                            multicast_path_min.add_node(n, vnf=[])
                    if vnf not in list(v[0] for v in multicast_path_min.nodes[i]['vnf']): 
                        G_min.nodes[i]['mem_capacity'] -= 1

                    # Processing data with transcoder
                    if "t" in vnf: # This vnf is transcoder
                        output_q = dsts[dst]
                        update_data_rate = exp_set.cal_transcode_bitrate(min_data_rate[dst][-1][2], min_data_rate[dst][-1][1], output_q)
                        min_data_rate[dst].append((i,output_q,update_data_rate))
                    else:
                        min_data_rate[dst].append((i,min_data_rate[dst][-1][1],min_data_rate[dst][-1][2])) 
                    
                    # If it has placed vnf, there don't need to add edge again.
                    if place_flag != 1:
                        # Add edge to connect node
                        if find_distance == 1:
                            if update_shortest_path_set[dst][-1] != i:
                                update_shortest_path_set[dst].append(i)
                            
                            e = (last_node, i)
                            # If previous_node to i don't have path then build it
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, min_data_rate[dst][-2], min_data_rate)
                        else:   
                            for j in range(len(shortest_path)-1):
                                if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                    update_shortest_path_set[dst].append(shortest_path[j+1])
                                if shortest_path[j] not in multicast_path_min:
                                    multicast_path_min.add_node(shortest_path[j], vnf=[])
                                e = (shortest_path[j], shortest_path[j+1])
                                if j != 0:
                                    min_data_rate[dst].insert(-2, (shortest_path[j],min_data_rate[dst][-1][1],min_data_rate[dst][-1][2]))
                                    #min_data_rate[dst].append((shortest_path[j],min_data_rate[dst][-1][1],min_data_rate[dst][-1][2]))
                                Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, min_data_rate[dst][-2], min_data_rate)

                    is_process_data = Graph.add_new_processing_data(G_min, multicast_path_min, update_shortest_path_set, dst, i, vnf, min_data_rate[dst][-2], min_data_rate)

                    if is_process_data == True:
                        index_sfc_min[dst]['index'] += 1
                        index_sfc_min[dst]['place_node'].append(i)                        
                        place_flag = 1
                    else:
                        G_min = copy.deepcopy(tmp_G)
                        multicast_path_min = copy.deepcopy(tmp_path)
                        update_shortest_path_set = copy.deepcopy(tmp_path_set)
                        min_data_rate = copy.deepcopy(tmp_data_rate)
                        place_flag = 2
                        break

                    if index_sfc_min[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                        break
                    vnf = sfc[dst][index_sfc_min[dst]['index']+1]

                if index_sfc_min[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                    break

    # print('place nearest')
    #Graph.printGraph(multicast_path_min, pos, 'tmp', service, 'data_rate')

    # Construct the path from the last_placement_node to dst and ignore bandwidth constraint
    for d in sort_dsts:
        dst = d[0]
        if dst not in missing_vnf_dsts:
            continue
        last_node = update_shortest_path_set[dst][-1]
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, dst, weight='weight')
        
        for i,j in enumerate(shortest_path):
            if i == 0:
                continue
            if j not in multicast_path_min:
                multicast_path_min.add_node(j, vnf=[])
            e = (last_node, j)
            update_shortest_path_set[dst].append(j)
            if j != dst:
                min_data_rate[dst].append((j,min_data_rate[dst][-1][1],min_data_rate[dst][-1][2]))
            index = len(update_shortest_path_set[dst]) - 2
            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, min_data_rate[dst][index], min_data_rate)
            last_node = j

    # print('link to dst')
    # Graph.printGraph(G_min, pos, 'G_min_update', service, 'bandwidth')
    # Graph.printGraph(multicast_path_min, pos, 'path_update', service, 'data_rate')

    # Graph.printGraph(G_new, pos, 'G_new', service, 'weight')
    # Graph.printGraph(G_metric_closure, pos, 'G_metric', service, 'distance')
    # Graph.printGraph(G_keyNode, pos, 'G_key', service, 'weight')
    # Graph.printGraph(T, pos, 'T', service, 'weight')

    # print(min_data_rate)
    # print(index_sfc_min)
    # print('======== nodes =========')
    # print(multicast_path_min.nodes(data=True))
    # print('======== edges =========')
    # print(multicast_path_min.edges(data=True))
    # print('\n------------')

    '''
    Check whether each path in multicast_path satisfy the link transmission rate requirement.
    If the single-path solution is infeasible, it will be extended to the multipath solution.
    '''
    G_final = copy.deepcopy(G)
    for m in multicast_path_min:
        G_final.nodes[m]['vnf'].extend(multicast_path_min.nodes[m]['vnf'])

    path_final = copy.deepcopy(multicast_path_min)
    path_final.remove_edges_from(path_final.edges())

    final_data_rate = dict()

    if min(list(G_min.edges[e]['bandwidth'] for e in G_min.edges())) >= 0:
        final_data_rate = copy.deepcopy(min_data_rate)
        #print('J = 1')
        # print(update_shortest_path_set)
        # print(final_data_rate)
        # print(index_sfc_min)
        return (G_min, multicast_path_min, sfc, update_shortest_path_set)

    # Record all path from src to dst with its ordered nodes
    # final_path_set = {dst: [the ordered nodes of shortest path]}
    final_path_set = dict()
    for d in sort_dsts:
        dst = d[0]
        final_path_set[dst] = [src]
        final_data_rate[dst] = [(src, best_quality, quality_list[best_quality])]
        dst_path = update_shortest_path_set[dst]

        last_vnf_node_index = 0
        isRebuild = 0

        for i in range(len(dst_path)-1):
            success_flag = False

            # Check if edges of multicast_path have enough resource to transmission data.
            # If enough, copy the original path data to final path.
            e = (dst_path[i], dst_path[i+1])
            if G_min.edges[e]['bandwidth'] <= 0:
                isRebuild = 1

            if dst_path[i+1] not in index_sfc_min[dst]['place_node'] and i+1 < len(dst_path):
                continue
            
            # If dst_path[i+1] is the node placing vnf
            tmp_path_set = dst_path[last_vnf_node_index:(i+1)]
            tmp_path_set.append(dst_path[i+1])

            if isRebuild == 0:
                for j in range(len(tmp_path_set)-1):
                    e = (tmp_path_set[j], tmp_path_set[j+1])
                    final_path_set[dst].append(tmp_path_set[j+1])
                    
                    Graph.add_new_edge(G_final, path_final, final_path_set, dst, e, min_data_rate[dst][last_vnf_node_index], min_data_rate)
                    final_data_rate[dst].append((dst_path[i], min_data_rate[dst][last_vnf_node_index][1], min_data_rate[dst][last_vnf_node_index][2]))
                last_vnf_node_index = i+1
                continue

            for tmp_n in tmp_path_set:
                if tmp_n != src and tmp_n not in dst_list and tmp_n in path_final.nodes() and path_final.nodes[tmp_n]['vnf'] == []:
                    path_final.remove_node(tmp_n)

            # Not enough, start searching candidate path.
            tree_list = list(nx.all_simple_paths(G, source=dst_path[last_vnf_node_index], target=dst_path[i+1]))
            tree_list_num = len(tree_list)

            # Sort all candidate paths in a descending order 
            # based on the amount of residual transmission resource.
            tree_resource = dict()
            for k in range(tree_list_num):
                tree_resource[k] = 0
                for m in range(len(tree_list[k])-1):
                    tree_resource[k] += G.edges[tree_list[k][m],tree_list[k][m+1]]['bandwidth']
            tree_resource_sort = sorted(tree_resource.items(), key=lambda x:x[1], reverse=True)
            
            # Build tree with sorted tree
            for j in range(tree_list_num):
                tmp_path = tree_list[tree_resource_sort[j][0]]

                min_bandwidth = sys.maxsize
                for m in range(len(tmp_path)-1):
                    min_bandwidth = min(G_final.edges[tmp_path[m], tmp_path[m+1]]['bandwidth'], min_bandwidth)
                
                # Check if edges of tmp_path have enough resource to transmission data
                data_rate = (min_data_rate[dst][last_vnf_node_index][1], min_data_rate[dst][last_vnf_node_index][2])
                if min_bandwidth > data_rate[1]:
                    
                    for m in tmp_path:
                        if m not in path_final:
                            path_final.add_node(m, vnf=[])
                    for m in range(len(tmp_path)-1):
                        final_path_set[dst].append(tmp_path[m+1])
                        
                        e = (tmp_path[m],tmp_path[m+1])

                        # allocate_data_rate = data_rate[1] / (j+1) # Averagely allocate data rate to each tree
                        tmp_data_rate = (tmp_path[m], data_rate[0], data_rate[1])
                        Graph.add_new_edge(G_final, path_final, final_path_set, dst, e, tmp_data_rate, final_data_rate)
                        final_data_rate[dst].append(tmp_data_rate)
                    
                    tmp_data_rate = (tmp_path[-1], final_data_rate[dst][-1][1], final_data_rate[dst][-1][2])
                    final_data_rate[dst].append(tmp_data_rate)
                    
                    success_flag = True
                    break
            
            if success_flag == True:
                isRebuild = 0
                last_vnf_node_index = i+1
                break
            
        if isRebuild == 1:
            #print("infeasible solution")
            # print(final_path_set)
            # print(final_data_rate)
            # print(index_sfc_min)
            # print('----------')
            # print(path_final.nodes(data=True))
            # print(path_final.edges(data=True))
            return (G, nx.Graph(), [], {})
        else:
            tmp_path_set = dst_path[last_vnf_node_index:(i+1)]
            tmp_path_set.append(dst_path[i+1])
            for m in range(len(tmp_path_set)-1):
                e = (tmp_path_set[m], tmp_path_set[m+1])
                #if m+1 != dst:
                final_path_set[dst].append(tmp_path_set[m+1])
                Graph.add_new_edge(G_final, path_final, final_path_set, dst, e, min_data_rate[dst][-1], min_data_rate)
                final_data_rate[dst].append((dst_path[-2], min_data_rate[dst][-1][1], min_data_rate[dst][-1][2]))
            if final_path_set[dst][-1] != dst:
                final_path_set[dst].append(dst)

    # Graph.printGraph(G_final, pos, 'G_final', service, 'bandwidth')
    # Graph.printGraph(path_final, pos, 'path_final', service, 'data_rate')
    # print(final_path_set)
    # print(final_data_rate)
    # print(index_sfc_min)

    return (G_final, path_final, sfc, final_path_set)