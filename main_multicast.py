'''
分組之後，針對同一組的人決定發送品質、最大需要多少個轉碼器
假設有三種品質，那最多需要兩個轉碼器。
接著對所有 src 到 dst 找最短路徑，找到最簡單的 tree
對 tree 做 greedy 找放置轉碼器的位置
這樣就找到每組的部署轉碼器位置及路由路徑

接著找最近的 group
對這兩個組找最好的品質，並對這兩個組的上游找可以放轉碼器的部分
'''

import copy
import sys
import networkx as nx

import Graph
import experience_setting as exp_set
import benchmark_JPR

def search_multipath(G, service, quality_list):
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
    
    video_type.reverse()

    best_qid = video_type.index(best_quality)
    
    # If dsts need different quality of video then place a transcoder
    if max_transcoder_num > 0: 
        for d in dst_list:
            for q_id in range(best_qid, len(video_type)):
                q = video_type[q_id]
                if dsts[d] != q:
                    sfc[d].append('t_'+str(q_id)+str(q_id+1))
                else:
                    break

    # print('--- environment ---')
    # print('src = ', src, ", dst = ", dsts)
    # print(sort_dsts)
    # print("sfc = ", sfc)

    G_min = copy.deepcopy(G)
    multicast_path_min = nx.Graph()
    # Record the satisfied vnf now and place node
    index_sfc = dict((d,{'index': -1, 'place_node':[], 'place_id':[]}) for d in dst_list)
    # Record all path from src to dst with its ordered nodes
    shortest_path_set = {}

    source_data_size = quality_list[best_quality]

    # Reocrd the current quality send to dst
    data_rate = dict()
    for d in dst_list:
        data_rate[d] = [(src, best_quality, source_data_size)]

    failed_dsts = list()

    # Find shortest path
    for d in sort_dsts:
        dst = d[0]
        
        # Build auxiliary graph of enough transimission, placement resources.
        remove_edges = list((n1,n2) for n1,n2,dic in G_min.edges(data=True) if dic['bandwidth'] < source_data_size)
        G_min.remove_edges_from(remove_edges)

        try:
            shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, src, dst, weight='data_rate')
        except nx.NetworkXNoPath:
            failed_dsts.append(d)
            sort_dsts.remove(d)
            continue
            
        shortest_path_set[dst] = [src]

        # Update nodes of multicast_path with nodes of shortest_path
        # node_attr = {node: [placement VNF]}
        node_attr = {}
        for m in shortest_path:
            if m not in multicast_path_min:
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
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                else: # Find node to place vnf
                    vnf = sfc[dst][index_sfc[dst]['index']+1]
                    vnf_find = vnf
                    if vnf[0] == 't':
                        vnf_find = "t"

                    # There is vnf instance on node j
                    if vnf_find in list(v[0][0] for v in G_min.nodes[j]['vnf']):
                        # Processing data with transcoder
                        if "t" in vnf: # This vnf is transcoder
                            output_q = video_type[int(vnf[-1])]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((j,output_q,update_data_rate))
                        else:
                            data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                        
                        is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, shortest_path_set, dst, i, vnf, data_rate[dst][-2], data_rate)
                        
                        if is_process_data == True:
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)
                            index_sfc[dst]['place_id'].append(i)
                            continue # A node place one vnf, if successfully placing vnf on node, then go to next node.
                        else: 
                            # Because compute capacity not enough, can't processing data.
                            data_rate[dst].pop(-1)
                            data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            continue

                    # Check if the node has enough resource to place instance
                    if G_min.nodes[j]['mem_capacity'] > 0: # memory capacity
                        # add new vnf instance
                        is_initialize = False
                        if vnf_find not in list(v[0][0] for v in G_min.nodes[j]['vnf']): 
                            G_min.nodes[j]['mem_capacity'] -= 1
                            is_initialize = True

                        # Processing data with transcoder
                        if "t" in vnf: # This vnf is transcoder
                            output_q = video_type[int(vnf[-1])]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((j,output_q,update_data_rate))
                        else:
                            data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                        
                        is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, shortest_path_set, dst, i, vnf, data_rate[dst][-2], data_rate)
                        
                        if is_process_data == True:
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)
                            index_sfc[dst]['place_id'].append(i)
                        else:
                            # Because compute capacity not enough, can't processing data.
                            data_rate[dst].pop(-1)
                            data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            if is_initialize == True:
                                G_min.nodes[j]['mem_capacity'] += 1
                    
                if data_rate[dst][-1][0] != j:
                    data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

            e = (shortest_path[i],shortest_path[i+1])
            Graph.add_new_edge(G_min, multicast_path_min, shortest_path_set, dst, e, data_rate[dst][-1], data_rate)

    #
    # A corrective subroutine that places the missing NF instances 
    # on the closest NFV node from the multicast topology.
    #
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
        place_flag = 0 # whether placing VNF or not (share or initiate), 0: not place, 1: place, 2: the node can't place
        
        while index_sfc[dst]['index'] < len(sfc[dst])-1:
            last_node = update_shortest_path_set[dst][-1]
            if place_flag != 1:
                find_distance += 1
                if find_distance >= len(G.edges()):
                    print('cannot find path')
                    failed_dsts = sort_dsts
                    return (G, nx.Graph(), list(), dict(), failed_dsts, failed_dsts)
            else:
                find_distance = 1
                place_flag = 0

            alternate_nodes_len = nx.single_source_shortest_path_length(G_min, last_node, find_distance)
            alternate_nodes = list(n for n in alternate_nodes_len)

            # Find the node with distance(<= find_distance) between last_node to place the unsatisfied VNF.
            # If find the node to place, then restart to find another node to place next VNF.
            for i in alternate_nodes:
                if place_flag == 1:
                    break

                # If find_distance > 1, the node that distance = 1 have been searched
                # it don't have to consider the node again
                if find_distance > 1 and alternate_nodes_len[i] <= find_distance-1:
                    continue

                last_node = update_shortest_path_set[dst][-1]
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, i, weight='data_rate')
                
                vnf = sfc[dst][index_sfc[dst]['index']+1]
                vnf_find = vnf
                if vnf[0] == 't':
                    vnf_find = "t"

                # Don't place on last_node, src, and dst
                if i == last_node or i == src or i == dst:
                    continue

                # There are vnf instance on node i
                if vnf_find in list(v[0][0] for v in G_min.nodes[i]['vnf']):
                    tmp_G = copy.deepcopy(G_min)
                    tmp_path = copy.deepcopy(multicast_path_min)
                    tmp_path_set = copy.deepcopy(update_shortest_path_set)
                    tmp_data_rate = copy.deepcopy(data_rate)
                    
                    # Processing data with transcoder
                    if "t" in vnf: # This vnf is transcoder
                        output_q = video_type[int(vnf[-1])]
                        update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                        data_rate[dst].append((i,output_q,update_data_rate))
                    else:
                        data_rate[dst].append((i,data_rate[dst][-1][1],data_rate[dst][-1][2])) 

                    # Add edge to connect node
                    if find_distance == 1:
                        if update_shortest_path_set[dst][-1] != i:
                            update_shortest_path_set[dst].append(i)
                        if i not in multicast_path_min:
                            multicast_path_min.add_node(i, vnf=[])
                        
                        e = (last_node, i)
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                    else:
                        for j in range(len(shortest_path)-1):
                            if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                update_shortest_path_set[dst].append(shortest_path[j+1])
                            if shortest_path[j] not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            if j != 0:
                                data_rate[dst].insert(-2, (shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                                #data_rate[dst].append((shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                
                    is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, update_shortest_path_set, dst, len(update_shortest_path_set[dst])-1, vnf, data_rate[dst][-2], data_rate)

                    if is_process_data == True:
                        index_sfc[dst]['index'] += 1
                        index_sfc[dst]['place_node'].append(i)
                        index_sfc[dst]['place_id'].append(len(update_shortest_path_set[dst])-1)
                        place_flag = 1
                    else:
                        G_min = copy.deepcopy(tmp_G)
                        multicast_path_min = copy.deepcopy(tmp_path)
                        update_shortest_path_set = copy.deepcopy(tmp_path_set)
                        data_rate = copy.deepcopy(tmp_data_rate)
                        place_flag = 2
            
                    continue

                    # if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                    #     break
                    # vnf = sfc[dst][index_sfc[dst]['index']+1]

                if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                    break

                # Check if node has enough resource to place instance
                if G_min.nodes[i]['mem_capacity'] >= 0: # memory capacity
                    if i not in multicast_path_min:
                        multicast_path_min.add_node(i, vnf=[])
                    
                    tmp_G = copy.deepcopy(G_min)
                    tmp_path = copy.deepcopy(multicast_path_min)
                    tmp_path_set = copy.deepcopy(update_shortest_path_set)
                    tmp_data_rate = copy.deepcopy(data_rate)

                    for n in multicast_path_min.nodes:
                        if 'vnf' not in multicast_path_min.nodes[n]:
                            multicast_path_min.add_node(n, vnf=[])
                    if vnf_find not in list(v[0][0] for v in G_min.nodes[i]['vnf']): 
                        G_min.nodes[i]['mem_capacity'] -= 1

                    # Processing data with transcoder
                    if "t" in vnf: # This vnf is transcoder
                        output_q = video_type[int(vnf[-1])]
                        update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                        data_rate[dst].append((i,output_q,update_data_rate))
                    else:
                        data_rate[dst].append((i,data_rate[dst][-1][1],data_rate[dst][-1][2])) 
                    
                    # Add edge to connect node
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
                            if shortest_path[j] not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            if j != 0:
                                data_rate[dst].insert(-2, (shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                                #data_rate[dst].append((shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)

                    is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, update_shortest_path_set, dst, len(update_shortest_path_set[dst])-1, vnf, data_rate[dst][-2], data_rate)

                    if is_process_data == True:
                        index_sfc[dst]['index'] += 1
                        index_sfc[dst]['place_node'].append(i)
                        index_sfc[dst]['place_id'].append(len(update_shortest_path_set[dst])-1)                      
                        place_flag = 1
                    else:
                        G_min = copy.deepcopy(tmp_G)
                        multicast_path_min = copy.deepcopy(tmp_path)
                        update_shortest_path_set = copy.deepcopy(tmp_path_set)
                        data_rate = copy.deepcopy(tmp_data_rate)
                        place_flag = 2
                        continue

                    if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                        break
                    vnf = sfc[dst][index_sfc[dst]['index']+1]
                    vnf_find = vnf
                    if vnf[0] == 't':
                        vnf_find = "t"

                if index_sfc[dst]['index'] >= len(sfc[dst])-1: # finish sfc
                    break

    # Construct the path from the last_placement_node to dst and ignore bandwidth constraint
    for d in sort_dsts:
        dst = d[0]
        if dst not in missing_vnf_dsts:
            continue
        last_node = update_shortest_path_set[dst][-1]
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, dst, weight='data_rate')
        
        for i,j in enumerate(shortest_path):
            if i == 0:
                continue
            if j not in multicast_path_min:
                multicast_path_min.add_node(j, vnf=[])
            e = (last_node, j)
            update_shortest_path_set[dst].append(j)
            if j != dst:
                data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
            
            index = len(update_shortest_path_set[dst]) - 2
            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][index], data_rate)
            last_node = j

    # print('=======update========')
    # print(update_shortest_path_set)
    # print(index_sfc)
    # print(data_rate)
    # print('===============')
    final_sfc = dict()
    for d in sort_dsts:
        dst = d[0]
        final_sfc[dst] = list()
        i = 0
        for v in sfc[dst]:
            final_sfc[dst].append([v,index_sfc[dst]['place_node'][i],index_sfc[dst]['place_id'][i]])
            i += 1

    return (G_min, multicast_path_min, update_shortest_path_set, final_sfc, data_rate, sort_dsts, failed_dsts)


def merge_group(G, G_min, src, quality_list, all_group_result):
    failed_group = list()
    all_group = list()
    for g in all_group_result:
        if nx.classes.function.is_empty(g[1]):
            failed_group.extend(g[-1])
        else:
            all_group.append((g[1],g[2],g[3],g[4],g[5]))
            if len(g[-1]) != 0:
                failed_group.extend(g[-1])
    
    # print(all_group)
    # print(failed_group)
    
    sort_group = sorted(all_group, key=lambda g: Graph.count_instance(g[0]), reverse=True)

    main_group = sort_group[0]
    #for i in range(1, len(sort_group)):

    new_path_info = merge_path(G_min, quality_list, main_group)

    #merge_group = sort_group[1]
    G_new = update_graph(G, new_path_info, main_group[4])

    # print(main_group)

    # print(all_group_result)
    return G_new
    
# Merge path from src to dst
def merge_path(G_min, quality_list, group_info):
    path_set = group_info[1]
    sfc = group_info[2]
    data_rate = group_info[3]
    reverse_sort = group_info[4]
    reverse_sort.reverse()

    # print('------------')
    # print(path_set)
    # print(sfc)
    # print(data_rate)
    # print('-----------')

    video_type = [q for q in quality_list]
    video_type.reverse()

    ### 怎麼決定誰要跟誰併，以及所有路徑的合併結束條件為何？
    main_dst = reverse_sort[0][0]
    src = path_set[main_dst][0]
    is_merge = False
    for i in range(1,len(reverse_sort)):
        if is_merge == True:
            break
        dst = reverse_sort[i][0]

        # First last_node is src
        last_node_id_main = 0 
        last_node_id_dst = 0
        
        for j,v in enumerate(sfc[dst]):
            # If the vnfs place on different node, then try to merge it.
            if v[1] != sfc[main_dst][j][1]:
                shortest_path_2vnf = nx.algorithms.shortest_paths.dijkstra_path(G_min, sfc[main_dst][j][1], v[1], weight='data_rate')
                
                original_cost = cal_cost_origin(last_node_id_main, j, sfc[main_dst], data_rate[main_dst]) + cal_cost_origin(last_node_id_dst, j, sfc[dst], data_rate[dst])
                min_cost = original_cost
                min_node = -1

                if j < len(sfc[main_dst])-1:
                    next_vnf_main = sfc[main_dst][j+1][1]
                    next_vnf_main_id = sfc[main_dst][j+1][2]
                else:
                    next_vnf_main = main_dst
                    next_vnf_main_id = -1
                
                if j < len(sfc[dst])-1:
                    next_vnf = sfc[dst][j+1][1]
                    next_vnf_id = sfc[dst][j+1][2]
                else:
                    next_vnf = dst
                    next_vnf_id = -1
                
                # Iterative all node between v[1] and sfc[main_dst][j][1],
                # calculate their cost, and get the node has minimim cost to merge two path.
                # 想一下如何update資料，及更新路徑（重建嗎）
                for m in shortest_path_2vnf:
                    if m == src or m == main_dst or m == dst:
                        continue
                    trans_cost_main = nx.dijkstra_path_length(G_min, path_set[main_dst][last_node_id_main], m, weight='data_rate') * data_rate[main_dst][last_node_id_main][2] + nx.dijkstra_path_length(G_min, m, next_vnf_main, weight='data_rate') * data_rate[main_dst][sfc[main_dst][j][2]][2]
                    trans_cost_dst = nx.dijkstra_path_length(G_min, path_set[dst][last_node_id_dst], m, weight='data_rate') * data_rate[dst][last_node_id_dst][2] + nx.dijkstra_path_length(G_min, m, next_vnf, weight='data_rate') * data_rate[dst][v[2]][2]

                    vnf_find = v[0]
                    if vnf_find[0] == 't':
                        vnf_find = "t"

                    total_cost = trans_cost_main + trans_cost_dst + data_rate[main_dst][last_node_id_main][2] + data_rate[dst][last_node_id_dst][2]
                    
                    # There aren't vnf instance on node i
                    if vnf_find not in list(x[0][0] for x in G_min.nodes[m]['vnf']):
                        total_cost += 1

                    if total_cost < min_cost:
                        min_cost = total_cost
                        min_node = m

                # Merge path and update information 
                if min_node != -1:
                    is_merge = True
                    # Update main_dst
                    new_path_main_before = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[main_dst][last_node_id_main], min_node, weight='data_rate')[:-1]
                    new_path_main_after = nx.algorithms.shortest_paths.dijkstra_path(G_min, min_node, next_vnf_main, weight='data_rate')[:-1]
                    if next_vnf_main_id == -1:
                        new_path_main_after.append(main_dst)
                        path_set[main_dst][last_node_id_main:] = new_path_main_before + new_path_main_after
                    else:
                        path_set[main_dst][last_node_id_main:next_vnf_main_id] = new_path_main_before + new_path_main_after

                    input_data = data_rate[main_dst][last_node_id_main][2]
                    output_data = data_rate[main_dst][sfc[main_dst][j][2]][2]
                    if next_vnf_main_id == -1:
                        del data_rate[main_dst][last_node_id_main:]
                    else:
                        del data_rate[main_dst][last_node_id_main:next_vnf_main_id]
                    for n in new_path_main_before:
                        data_rate[main_dst].insert(last_node_id_main, (n, video_type[int(sfc[main_dst][j][0][-2])], input_data))
                        last_node_id_main += 1

                    sfc[main_dst][j] = [sfc[main_dst][j][0], min_node, last_node_id_main]
                    for f in range(j+1,len(sfc[main_dst])):
                        sfc[main_dst][f][2] = len(new_path_main_after) + last_node_id_main + (f - j) - 1
                    
                    for n in new_path_main_after:
                        data_rate[main_dst].insert(last_node_id_main, (n, video_type[int(sfc[main_dst][j][0][-1])], output_data))
                        last_node_id_main += 1

                    # Update dst
                    new_path_dst_before = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst][last_node_id_dst], min_node, weight='data_rate')[:-1]
                    new_path_dst_after = nx.algorithms.shortest_paths.dijkstra_path(G_min, min_node, next_vnf, weight='data_rate')[:-1]
                    if next_vnf_id == -1:
                        new_path_dst_after.append(dst)
                        path_set[dst][last_node_id_dst:] = new_path_dst_before + new_path_dst_after
                    else:
                        path_set[dst][last_node_id_dst:next_vnf_id] = new_path_dst_before + new_path_dst_after

                    input_data = data_rate[dst][last_node_id_dst][2]
                    output_data = data_rate[dst][v[2]][2]
                    if next_vnf_id == -1:
                        del data_rate[dst][last_node_id_dst:]
                    else:
                        del data_rate[dst][last_node_id_dst:next_vnf_id]
                    for n in new_path_dst_before:
                        data_rate[dst].insert(last_node_id_dst, (n, video_type[int(sfc[dst][j][0][-2])], input_data))
                        last_node_id_dst += 1

                    sfc[dst][j] = [sfc[dst][j][0], min_node, last_node_id_dst]
                    for f in range(j+1,len(sfc[dst])):
                        sfc[dst][f][2] = len(new_path_dst_after) + last_node_id_dst + (f - j) - 1

                    for n in new_path_dst_after:
                        data_rate[dst].insert(last_node_id_dst, (n, video_type[int(sfc[dst][j][0][-1])], output_data))
                        last_node_id_dst += 1

                last_node_id_main = sfc[main_dst][j][2]
                last_node_id_dst = sfc[dst][j][2]

    return (path_set, sfc, data_rate)

# Calculate total cost from src_vnf to dst_vnf.
def cal_cost_origin(last_node_id, index, vnf_info, data_rate):
    if index < len(vnf_info)-1:
        dst_vnf_id = vnf_info[index+1][2]
    else:
        dst_vnf_id = len(data_rate)
    
    # Transimission cost
    trans_cost = 0
    for i in range(last_node_id,dst_vnf_id):
        trans_cost += data_rate[i][2]

    ### 如何確定共享的 vnf placing cost
    return trans_cost + data_rate[last_node_id][2] + 1
        
# Use new_path_info to create new path on G        
def update_graph(G, new_path_info, sort_dsts):
    path_set = new_path_info[0]
    sfc = new_path_info[1]
    data_rate = new_path_info[2]
    
    # print(sort_dsts)
    # print('=============')
    # print(path_set)
    # print(sfc)
    # print(data_rate)
    # print('=============')

    G_min = copy.deepcopy(G)
    multicast_path = nx.Graph()

    tmp_list = dict()
    for d in sort_dsts:
        dst = d[0]
        tmp_list[dst] = [path_set[dst][0]]

        sfc_index = 0
        for i in range(1,len(path_set[dst])):
            j = path_set[dst][i]
            tmp_list[dst].append(j)
            if j not in multicast_path:
                multicast_path.add_node(j, vnf=[])

            # placing vnf
            
            if sfc_index < len(sfc[dst]) and i == sfc[dst][sfc_index][2]:
                #print(dst, sfc_index, sfc[dst][sfc_index], sfc[dst])
                vnf = sfc[dst][sfc_index][0]
                vnf_find = vnf
                if vnf[0] == 't':
                    vnf_find = "t"

                # There is vnf instance on node j
                if vnf_find not in list(v[0][0] for v in G_min.nodes[j]['vnf']):
                    G_min.nodes[j]['mem_capacity'] -= 1
                is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path, tmp_list, dst, i, vnf, data_rate[dst][i-1], data_rate)
                sfc_index += 1

            # connect edge
            e = (path_set[dst][i-1], path_set[dst][i])

            Graph.add_new_edge(G_min, multicast_path, tmp_list, dst, e, data_rate[dst][i-1], data_rate)

    #print(multicast_path.nodes(data=True))
    return G_min