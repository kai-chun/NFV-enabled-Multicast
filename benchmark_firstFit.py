import networkx as nx
import copy

import Graph
import experience_setting as exp_set
import benchmark_JPR

'''
Use Dijkstra algorithm find the path from src to each dst.
And placing ordered VNF from path head. 
A VNF is dedicated for one dst, not share with other dsts.
'''
def search_multipath(G, pos, service, quality_list):
    
    video_type = [q for q in quality_list]

    src = service[0]
    dsts = service[1]
    sort_dsts = sorted(service[1].items(), key=lambda d: video_type.index(d[1]), reverse=True)

    dst_list = list(d for d in dsts)

    best_quality = service[3]

    sfc = service[2]
    require_quality = set(dsts[i] for i in dsts)
    max_transcoder_num = len(require_quality) - 1
    sort_quality = sorted(require_quality, key=lambda q: video_type.index(q), reverse=True)

    #Graph.printGraph(G, pos, 'tmp', service, 'bandwidth')

    # If dsts need different quality of video then place a transcoder
    if max_transcoder_num > 0: 
        for d in dst_list:
            if dsts[d] != best_quality:
                sfc[d].append('t_'+str(d))

    # print('--- environment ---')
    # print('src = ', src, ", dst = ", dsts)
    # print(sort_dsts)
    # print("sfc = ", sfc)

    G_min = copy.deepcopy(G)
    multicast_path_min = nx.Graph()
    # Record the satisfied vnf now and place node
    index_sfc = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list)
    # Record all path from src to dst with its ordered nodes
    shortest_path_set = {}

    # Reocrd the current quality send to dst
    data_rate = dict()
    for d in dst_list:
        data_rate[d] = [(src, best_quality, quality_list[best_quality])]

    # Find shortest path
    for d in sort_dsts:
        dst = d[0]
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, src, dst, weight='weight')
        shortest_path_set[dst] = [src]

        # Update nodes of tmp_multicast_path with nodes of shortest_path
        # node_attr = {node: [placement VNF]}
        node_attr = {}
        for m in shortest_path:
            if m not in multicast_path_min or len(multicast_path_min.nodes[m]) == 0:
                node_attr[m] = {'vnf': []}
            else:
                node_attr[m] = multicast_path_min.nodes[m]
        multicast_path_min.add_nodes_from(shortest_path)
        nx.set_node_attributes(multicast_path_min, node_attr)
        
        for i in range(len(shortest_path)-1):
            shortest_path_set[dst].append(shortest_path[i+1])

            # Don't place vnf on src and dst
            if shortest_path[i] != src and shortest_path[i] != dst:
                j = shortest_path[i]

                if index_sfc[dst]['index'] >= len(sfc[dst])-1:
                    # output_q = sort_quality[index_sfc[dst]['index']+1]
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

                # Find node to place vnf until finishing sfc
                while index_sfc[dst]['index'] < len(sfc[dst])-1:
                    vnf = sfc[dst][index_sfc[dst]['index']+1]

                    # Check if the node has enough resource to place instance
                    if G_min.nodes[j]['mem_capacity'] > 0: # memory capacity
                        if G_min.nodes[j]['com_capacity'] > data_rate[dst][-1][2]: # compute capacity
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)

                            if vnf not in multicast_path_min.nodes[j]['vnf']:
                                G_min.nodes[j]['mem_capacity'] -= 1
                            
                            G_min.nodes[j]['vnf'].append((vnf, data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            multicast_path_min.nodes[j]['vnf'].append((vnf, data_rate[dst][-1][1], data_rate[dst][-1][2]))
                            G_min.nodes[j]['com_capacity'] -= data_rate[dst][-1][2]

                            # Processing data with transcoder
                            if "t" in vnf: # This vnf is transcoder
                                output_q = dsts[dst]
                                update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                                data_rate[dst].append((j,output_q,update_data_rate))
                            else:
                                data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))  
                        else:
                            break
                    else:
                        break
                if data_rate[dst][-1][0] != j:
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

            e = (shortest_path[i],shortest_path[i+1])
            Graph.add_new_edge(G_min, multicast_path_min, shortest_path_set, dst, e, data_rate[dst][-1], data_rate)

    '''
    A corrective subroutine that places the missing NF instances 
    on the closest NFV node from the multicast topology.
    '''
    tmp = benchmark_JPR.update_path(sfc, index_sfc, sort_dsts, G, G_min, multicast_path_min, shortest_path_set, data_rate)
    missing_vnf_dsts = tmp[0]
    update_shortest_path_set = tmp[1]

    # Fill up the empty min_data_rate[dst] with best quality (initial date to send)
    for d in dst_list:
        if len(data_rate[d]) == 0:
            data_rate[d] = [(src, best_quality, quality_list[best_quality])]

    # Place vnf nearest the common path
    for d in sort_dsts:
        dst = d[0]
        if dst not in missing_vnf_dsts:
            continue

        find_distance = 0 # the length range of finding node away from last_node that placing VNF
        place_flag = 0 # whether placing VNF or not

        while index_sfc[dst]['index'] < len(sfc[dst])-1:
            last_node = update_shortest_path_set[dst][-1]
            if place_flag == 0:
                find_distance += 1
            else:
                find_distance = 1
                place_flag = 0
            alternate_nodes_len = nx.single_source_shortest_path_length(G_min, last_node, find_distance)
            alternate_nodes = list(n for n in nx.single_source_shortest_path_length(G_min, last_node, find_distance))

            for i in alternate_nodes:
                if place_flag == 1:
                    break
                # If find_distance > 1, the node that distance = 1 have been searched
                # it don't have to consider the node again
                if find_distance > 1 and alternate_nodes_len[i] == find_distance-1:
                    continue

                last_node = update_shortest_path_set[dst][-1]
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, i, weight='weight')
                
                vnf = sfc[dst][index_sfc[dst]['index']+1]
                if i == last_node or i == src or i == dst:
                    continue

                # Check if node has enough resource to place instance
                while G_min.nodes[i]['mem_capacity'] >= 0: # memory capacity

                    if G_min.nodes[i]['com_capacity'] > data_rate[dst][-1][2]: # compute capacity
                        if i not in multicast_path_min or len(multicast_path_min.nodes[i]) == 0:
                            multicast_path_min.add_node(i, vnf=[])

                        index_sfc[dst]['index'] += 1
                        index_sfc[dst]['place_node'].append(i)

                        if vnf not in multicast_path_min.nodes[i]['vnf']: 
                            G_min.nodes[i]['mem_capacity'] -= 1

                        G_min.nodes[i]['vnf'].append((vnf, data_rate[dst][-1][1],data_rate[dst][-1][2]))
                        multicast_path_min.nodes[i]['vnf'].append((vnf, data_rate[dst][-1][1], data_rate[dst][-1][2]))
                        G_min.nodes[i]['com_capacity'] -= data_rate[dst][-1][2]                  

                        # Processing data with transcoder
                        if "t" in vnf: # This vnf is transcoder
                            output_q = dsts[dst]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((i,output_q,update_data_rate))
                        else:
                            data_rate[dst].append((i,data_rate[dst][-1][1],data_rate[dst][-1][2]))  

                        if place_flag != 1:
                            if find_distance == 1:
                                if update_shortest_path_set[dst][-1] != i:
                                    update_shortest_path_set[dst].append(i)
                                
                                e = (last_node, i)
                                # If previous_node to i don't have path then build it
                                Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                            else:
                                for j in range(len(shortest_path)-1):
                                    if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                        update_shortest_path_set[dst].append(shortest_path[j+1])
                                    e = (shortest_path[j], shortest_path[j+1])
                                    if j != 0:
                                        data_rate[dst].insert(-2, (shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                                        #data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                                    Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)

                        place_flag = 1  

                        if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                            break
                        vnf = sfc[dst][index_sfc[dst]['index']+1]
                    else:
                        break
                    
                    if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                        break

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
            if j not in multicast_path_min or len(multicast_path_min.nodes[j]) == 0:
                multicast_path_min.add_node(j, vnf=[])
            e = (last_node, j)
            update_shortest_path_set[dst].append(j)
            if j != dst:
                data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
            
            index = len(update_shortest_path_set[dst]) - 2
            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][index], data_rate)
            last_node = j
    
    # print('===============')
    # print(update_shortest_path_set)
    # print(data_rate)
    # print(index_sfc)
    # print('===============')
    # print(multicast_path_min.edges(data=True))
    
    # Check if edges of multicast_path have enough resource to transmission data.
    min_bandwidth = min(list(dic['bandwidth'] for (n1,n2,dic) in G_min.edges(data=True)))
    if min_bandwidth < 0:
        return (G, nx.Graph())

    return (G_min, multicast_path_min)