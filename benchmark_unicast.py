import networkx as nx
import copy

import experience_setting as exp_set

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

    sfc = service[2]
    require_quality = set(dsts[i] for i in dsts)
    max_transcoder_num = len(require_quality) - 1

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
    unicast_path_min = nx.Graph()

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

        # Update nodes of tmp_unicast_path with nodes of shortest_path
        # node_attr = {node: [placement VNF]}
        node_attr = {}
        for m in shortest_path:
            if m not in unicast_path_min:
                node_attr[m] = {'vnf': []}
            else:
                node_attr[m] = unicast_path_min.nodes[m]
        unicast_path_min.add_nodes_from(shortest_path)
        nx.set_node_attributes(unicast_path_min, node_attr)

        for i in range(len(shortest_path)-1):
            shortest_path_set[dst].append(shortest_path[i+1])

            # Don't place vnf on src and dst
            if shortest_path[i] != src and shortest_path[i] != dst:
                j = shortest_path[i]

                if index_sfc[dst]['index'] >= len(sfc[dst])-1:
                    #output_q = sort_quality[index_sfc[dst]['index']+1]
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

                # Find node to place vnf until finishing sfc
                while index_sfc[dst]['index'] < len(sfc[dst])-1:
                    vnf = sfc[dst][index_sfc[dst]['index']+1]

                    # Check if the node has enough resource to place instance
                    if G_min.nodes[j]['mem_capacity'] > 0: # memory capacity
                        if G_min.nodes[j]['com_capacity'] > data_rate[dst][-1][2]: # compute capacity
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)

                            if vnf not in unicast_path_min.nodes[j]['vnf']:
                                G_min.nodes[j]['mem_capacity'] -= 1

                            G_min.nodes[j]['vnf'].append((vnf, data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            unicast_path_min.nodes[j]['vnf'].append((vnf, data_rate[dst][-1][1], data_rate[dst][-1][2]))
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
            add_new_edge(G_min, unicast_path_min, e, data_rate[dst][-1])

    '''
    A corrective subroutine that places the missing NF instances 
    on the closest NFV node from the unicast topology.
    '''
    missing_vnf_dsts = []
    for dst in index_sfc: # find all dst which missing vnf 
        if index_sfc[dst]['index'] < len(sfc[dst])-1:
            missing_vnf_dsts.append(dst)

    if len(missing_vnf_dsts) > 0:
        # Remove unnecessary edges: the path from last vnf node to dst
        update_shortest_path_set = copy.deepcopy(shortest_path_set)
        for dst in missing_vnf_dsts:
            last_place_node = index_sfc[dst]['place_node'][-1] if index_sfc[dst]['index'] > -1 else -1
            last_node_index = shortest_path_set[dst].index(last_place_node) if last_place_node != -1 else 0
            delete_node = update_shortest_path_set[dst][last_node_index+1:]
            for i in delete_node:
                if i in unicast_path_min.nodes() and unicast_path_min.nodes[i]['vnf'] != []:
                    delete_node.remove(i)
            unicast_path_min.remove_nodes_from(delete_node)
            if dst not in unicast_path_min.nodes():
                unicast_path_min.add_node(dst, vnf=[])
            del update_shortest_path_set[dst][last_node_index+1:]
            
        for dst in update_shortest_path_set:
            for i in range(len(data_rate[dst])-1,-1,-1):
                data = data_rate[dst][i]
                if data[0] not in update_shortest_path_set[dst]:
                    data_rate[dst].pop(i)
                if len(update_shortest_path_set[dst]) == 1:
                    data_rate[dst] = [(src, best_quality, quality_list[best_quality])]
                    break

        # Rebuilding the multicast path with update_shortest_path_set
        G_min.remove_edges_from(G_min.edges())
        for n1,n2,d in G.edges(data=True):
            G_min.add_edge(n1,n2,data_rate=d['data_rate'],bandwidth=d['bandwidth'])

        unicast_path_min.remove_edges_from(unicast_path_min.edges())
        tmp_list = dict()
        for d in sort_dsts:
            dst = d[0]
            tmp_list[dst] = []
            for i in range(len(update_shortest_path_set[dst])-1):
                e = (update_shortest_path_set[dst][i], update_shortest_path_set[dst][i+1])

                list_node = list(data_rate[dst].index(j) for j in data_rate[dst] if update_shortest_path_set[dst][i] == j[0])
                if len(list_node) == 0:
                    node_index = 0
                else:
                    node_index = max(list_node)
                tmp_list[dst].append(update_shortest_path_set[dst][i])
                add_new_edge(G_min, unicast_path_min, e, data_rate[dst][node_index])
    else:
        update_shortest_path_set = copy.deepcopy(shortest_path_set)
        tmp_list = dict()
        #print('no miss')

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
                        index_sfc[dst]['index'] += 1
                        index_sfc[dst]['place_node'].append(i)

                        if i not in unicast_path_min:
                            unicast_path_min.add_node(i, vnf=[])

                        if vnf not in unicast_path_min.nodes[i]['vnf']: 
                            G_min.nodes[i]['mem_capacity'] -= 1

                        # If haven't place vnf on this node, then add edge to connect it.
                        if place_flag != 1:
                            if find_distance == 1:
                                if update_shortest_path_set[dst][-1] != i:
                                    update_shortest_path_set[dst].append(i)
                            
                                e = (last_node, i)
                                # If previous_node to i don't have path then build it
                                add_new_edge(G_min, unicast_path_min, e, data_rate[dst][-1])
                            else:
                                for j in range(len(shortest_path)-1):
                                    if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                        update_shortest_path_set[dst].append(shortest_path[j+1])
                                    e = (shortest_path[j], shortest_path[j+1])
                                    add_new_edge(G_min, unicast_path_min, e, data_rate[dst][-1])

                        G_min.nodes[i]['vnf'].append((vnf, data_rate[dst][-1][1],data_rate[dst][-1][2]))
                        unicast_path_min.nodes[i]['vnf'].append((vnf, data_rate[dst][-1][1], data_rate[dst][-1][2]))
                        G_min.nodes[i]['com_capacity'] -= data_rate[dst][-1][2]
                        place_flag = 1

                        # Processing data with transcoder
                        if "t" in vnf: # This vnf is transcoder
                            output_q = dsts[dst]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((i,output_q,update_data_rate))
                        else:
                            data_rate[dst].append((i,data_rate[dst][-1][1],data_rate[dst][-1][2]))  
                    
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
            if j not in unicast_path_min:
                unicast_path_min.add_node(j, vnf=[])
            e = (last_node, j)
            update_shortest_path_set[dst].append(j)
            add_new_edge(G_min, unicast_path_min, e, data_rate[dst][-1])
            if j != dst:
                data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
            last_node = j

    # print('===============')
    # print(update_shortest_path_set)
    # print(data_rate)
    # print(index_sfc)
    # print('===============')
    # print(unicast_path_min.edges(data=True))

    # Check if edges of multicast_path have enough resource to transmission data.
    min_bandwidth = min(list(dic['bandwidth'] for (n1,n2,dic) in G_min.edges(data=True)))
    if min_bandwidth < 0:
        return (G, nx.Graph())

    return (G_min, unicast_path_min)




