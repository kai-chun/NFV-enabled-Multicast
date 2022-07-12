'''
分組之後，針對同一組的人決定發送品質、最大需要多少個轉碼器
假設有三種品質，那最多需要兩個轉碼器。
接著對所有 src 到 dst 找最短路徑，找到最簡單的 tree
對 tree 做 greedy 找放置轉碼器的位置
這樣就找到每組的部署轉碼器位置及路由路徑

接著找最近的 group
對這兩個組找最好的品質，並對這兩個組的上游找可以放轉碼器的部分
'''

### First-Fit & shortest path & merging

import copy
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time

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

    sfc = copy.deepcopy(service[2])
    require_quality = set(dsts[i] for i in dsts)
    max_transcoder_num = len(require_quality) - 1
    sort_quality = sorted(require_quality, key=lambda q: video_type.index(q), reverse=True)
    
    video_type.reverse()

    best_qid = video_type.index(best_quality)
    
    # If dsts need different quality of video then place a transcoder
    if max_transcoder_num > 0: 
        for d in dst_list:
            if dsts[d] != best_quality:
                sfc[d].append('t')

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
        G_tmp = copy.deepcopy(G_min)
        tmp_path = copy.deepcopy(multicast_path_min)
        tmp_path_set = copy.deepcopy(shortest_path_set)
        tmp_data_rate = copy.deepcopy(data_rate)
        tmp_index_sfc = copy.deepcopy(index_sfc)
        is_connect = True

        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, src, dst, weight='weight')
        
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
                    is_place = False

                    vnf_find = vnf
                    if vnf[0] == 't':
                        vnf_find = "t"

                    # There is vnf instance on node j
                    if vnf_find in list(v[0][0] for v in G_min.nodes[j]['vnf']):
                        # Processing data with transcoder
                        if "t" in vnf: # This vnf is transcoder
                            output_q = dsts[dst]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((j,output_q,update_data_rate))
                        else:
                            data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))
                        
                        is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, shortest_path_set, dst, i, vnf, data_rate[dst][-2], data_rate)
                        
                        if is_process_data == True:
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)
                            index_sfc[dst]['place_id'].append(i)
                            is_place = True # A node place one vnf, if successfully placing vnf on node, then go to next node.
                        else: 
                            # Because compute capacity not enough, can't processing data.
                            data_rate[dst].pop(-1)
                            data_rate[dst].append((j,data_rate[dst][-1][1],data_rate[dst][-1][2]))

                    # Check if the node has enough resource to place instance
                    if is_place == False and G_min.nodes[j]['mem_capacity'] > 0: # memory capacity
                        # add new vnf instance
                        is_initialize = False
                        if vnf_find not in list(v[0][0] for v in G_min.nodes[j]['vnf']): 
                            G_min.nodes[j]['mem_capacity'] -= 1
                            is_initialize = True

                        # Processing data with transcoder
                        if "t" in vnf: # This vnf is transcoder
                            output_q = dsts[dst]
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
            is_connect = Graph.add_new_edge_main(G_min, multicast_path_min, shortest_path_set, dst, e, data_rate[dst][-1], data_rate)
            if not is_connect: break

        if is_connect == False:
            G_min = copy.deepcopy(G_tmp)
            multicast_path_min = copy.deepcopy(tmp_path)
            shortest_path_set = copy.deepcopy(tmp_path_set)
            data_rate = copy.deepcopy(tmp_data_rate)
            index_sfc = copy.deepcopy(tmp_index_sfc)
            failed_dsts.append(d)

    # A corrective subroutine that places the missing NF instances 
    # on the closest NFV node from the multicast topology.
    for d in sort_dsts:
        if d[0] in list(f[0] for f in failed_dsts):
            index_sfc.pop(d[0],None)
    
    tmp = benchmark_JPR.update_path(sfc, index_sfc, sort_dsts, G, G_min, multicast_path_min, shortest_path_set, data_rate)
    missing_vnf_dsts = tmp[0]
    update_shortest_path_set = tmp[1]

    # Fill up the empty min_data_rate[dst] with best quality (initial date to send)
    for d in dst_list:
        if len(data_rate[d]) == 0 and d not in list(f[0] for f in failed_dsts):
            data_rate[d] = [(src, best_quality, quality_list[best_quality])]

    # Place vnf nearest the common path
    for d in sort_dsts:
        dst = d[0]
        if dst not in missing_vnf_dsts or d in list(f[0] for f in failed_dsts):
            continue

        find_distance = 0 # the length range of finding node away from last_node that placing VNF
        place_flag = 0 # whether placing VNF or not (share or initiate), 0: not place, 1: place, 2: the node can't place
        
        while index_sfc[dst]['index'] < len(sfc[dst])-1:
            last_node = update_shortest_path_set[dst][-1]
            if place_flag != 1:
                find_distance += 1
                if find_distance >= len(G.edges()):
                    print('cannot find path')
                    failed_dsts.append(d)
                    break
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
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, i, weight='weight')
                
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
                    is_connect = True
                    
                    # Processing data with transcoder
                    if "t" in vnf: # This vnf is transcoder
                        output_q = dsts[dst]
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
                        is_connect = Graph.add_new_edge_main(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                    else:
                        for j in range(len(shortest_path)-1):
                            if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                update_shortest_path_set[dst].append(shortest_path[j+1])
                            if shortest_path[j] not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            if j != 0:
                                data_rate[dst].insert(-1, (shortest_path[j],data_rate[dst][0][1],data_rate[dst][0][2]))
                            is_connect = Graph.add_new_edge_main(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)

                    is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, update_shortest_path_set, dst, len(update_shortest_path_set[dst])-1, vnf, data_rate[dst][-2], data_rate)

                    if is_process_data == True and is_connect == True:
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
                        output_q = dsts[dst]
                        update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                        data_rate[dst].append((i,output_q,update_data_rate))
                    else:
                        data_rate[dst].append((i,data_rate[dst][-1][1],data_rate[dst][-1][2])) 
                    
                    # Add edge to connect node
                    if find_distance == 1:
                        if update_shortest_path_set[dst][-1] != i:
                            update_shortest_path_set[dst].append(i)

                        e = (last_node, i)
                        is_connect = Graph.add_new_edge_main(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                    else:   
                        for j in range(len(shortest_path)-1):
                            if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                update_shortest_path_set[dst].append(shortest_path[j+1])
                            if shortest_path[j] not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            if j != 0:
                                data_rate[dst].insert(-1, (shortest_path[j],data_rate[dst][0][1],data_rate[dst][0][2]))
                            is_connect = Graph.add_new_edge_main(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)

                    is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path_min, update_shortest_path_set, dst, len(update_shortest_path_set[dst])-1, vnf, data_rate[dst][-2], data_rate)

                    if is_process_data == True and is_connect == True:
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
        if dst not in missing_vnf_dsts or d in list(f[0] for f in failed_dsts):
            continue
        last_node = update_shortest_path_set[dst][-1]

        # Build auxiliary graph of enough transimission, placement resources.
        G_tmp = copy.deepcopy(G_min)
        remove_edges = list((n1,n2) for n1,n2,dic in G_min.edges(data=True) if dic['bandwidth'] < data_rate[dst][-1][2])
        G_tmp.remove_edges_from(remove_edges)

        try:
            shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_tmp, last_node, dst, weight='weight')
        except:
            failed_dsts.append(d)
            pass

        if d in failed_dsts: continue
        
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

    for d in dst_list:
        if d in list(f[0] for f in failed_dsts):
            update_shortest_path_set.pop(d, None)
            data_rate.pop(d, None)
            sfc.pop(d, None)

    for d in failed_dsts:
        sort_dsts.remove(d)

    final_sfc = dict()
    for d in sort_dsts:
        dst = d[0]
        final_sfc[dst] = list()
        i = 0
        for v in sfc[dst]:
            final_sfc[dst].append([v,index_sfc[dst]['place_node'][i],index_sfc[dst]['place_id'][i]])
            i += 1

    return (G_min, multicast_path_min, update_shortest_path_set, final_sfc, data_rate, sort_dsts, failed_dsts, sfc)

def merge_group(G, G_min, src, quality_list, all_group_result, weight):
    # Tackle the information of each group
    failed_group = list()
    sort_group = list()
    for g in all_group_result:
        if nx.classes.function.is_empty(g[1]):
            # print("empty")
            failed_group.extend(g[6])
        else:
            sort_group.append([g[2],g[3],g[4],g[5]])
            if len(g[6]) != 0:
                failed_group.extend(g[6])
    
    G_merge = copy.deepcopy(G)
    for i in range(len(sort_group)):
        G_merge = update_graph(G_merge, sort_group[i])

    merge_group_info = copy.deepcopy(sort_group)
    

    # Calculate the cost of merge 2 groups, and choose the minimum cost merging strategy.
    is_finish = False
    while is_finish == False:
        cost_metric = dict()

        # Calculate original cost
        orig_cost_groups = Graph.cal_total_cost_normalize(G, G_merge, weight, True)

        for i in range(len(sort_group)):
            for j in range(i+1, len(sort_group)):
                if len(merge_group_info[i][3]) == 0 or len(merge_group_info[j][3]) == 0:
                    cost_metric[(i,j)] = [0,0,0]
                    continue

                # Calculate merging cost
                new_path_info_merge = add_merge_info(G_merge, src, merge_group_info, i, j, quality_list)
                G_tmp = copy.deepcopy(G)

                for m in range(len(new_path_info_merge)):
                    if new_path_info_merge[m][0] == []: continue
                    G_tmp = update_graph(G_tmp, new_path_info_merge[m])

                merge_cost_groups = Graph.cal_total_cost_normalize(G, G_tmp, weight, True)

                cost_metric[(i,j)] = [round(orig_cost_groups[0] - merge_cost_groups[0], 2), new_path_info_merge, G_tmp]
        
        if cost_metric == {}:
            is_finish = True
            # print("no merge")
            break
        
        merge_result = max(cost_metric, key=lambda k: cost_metric[k][0])
        if cost_metric[merge_result][0] <= 0:
            is_finish = True
            # print("no merge")
            break

        merge_group_info = copy.deepcopy(cost_metric[merge_result][1])
        G_merge = copy.deepcopy(cost_metric[merge_result][2])

    
    # print("before merge path", time.time()-pre_time)
    # print("merge group: ", Graph.cal_total_cost(G_merge, weight, True))

    G_final = copy.deepcopy(G_merge)
    final_path_info = copy.deepcopy(merge_group_info)
    # print("fail_before",len(failed_group))
    not_failed_dsts = []
    # Find path for failed dst again
    if len(failed_group) != 0:
        # print("fail")
        all_dst = dict()
        for i in range(len(final_path_info)):
            for d in final_path_info[i][-1]:
                all_dst[d[0]] = (i, d[1])

        for f_dst in failed_group:
            distance_list = {nx.shortest_path_length(G, f_dst[0], j):j for j in all_dst}
            distance_list = sorted(distance_list.items())
            for j in range(min(3,len(distance_list))):
                if f_dst in not_failed_dsts: break
                nearest_dst = distance_list[j][1]
            
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_final, nearest_dst, f_dst[0], weight='weight')
                # print(final_path_info[all_dst[nearest_dst][0]][0][nearest_dst])
                can_connect = True
                for i in range(len(shortest_path)-1):
                    e = (shortest_path[i],shortest_path[i+1])
                    if G_final.edges[e]['bandwidth'] < quality_list[f_dst[1]]:
                        can_connect = False
                        break
                if can_connect:
                    not_failed_dsts.append(f_dst)
                    group_id = all_dst[nearest_dst][0]

                    if f_dst[1] == all_dst[nearest_dst][1]: # need same video quality
                        final_path_info[group_id][0][f_dst[0]] = copy.deepcopy(final_path_info[group_id][0][nearest_dst] + shortest_path[1:])
                        final_path_info[group_id][1][f_dst[0]] = copy.deepcopy(final_path_info[group_id][1][nearest_dst])
                        final_path_info[group_id][2][f_dst[0]] = copy.deepcopy(final_path_info[group_id][2][nearest_dst])
                        for n in shortest_path[:-1]:
                            final_path_info[group_id][2][f_dst[0]].append((n,f_dst[1],quality_list[f_dst[1]]))
                        final_path_info[group_id][3].append(f_dst)
                    else: # need different video quality
                        final_path_info.append([{},{},{},[]])
                        final_path_info[-1][0][f_dst[0]] = copy.deepcopy(final_path_info[group_id][0][nearest_dst] + shortest_path[1:])
                        final_path_info[-1][1][f_dst[0]] = []
                        final_path_info[-1][2][f_dst[0]] = []
                        for n in final_path_info[-1][0][f_dst[0]][:-1]:
                            final_path_info[-1][2][f_dst[0]].append((n,f_dst[1],quality_list[f_dst[1]]))
                        final_path_info[-1][3].append(f_dst)
                        # print("----")
                        # print(nearest_dst,all_dst[nearest_dst][1],final_path_info[group_id][0][nearest_dst])
                        # print(final_path_info[group_id][2][nearest_dst])
                        # print(f_dst[0],f_dst[1],final_path_info[-1][0][f_dst[0]])
                        # print(final_path_info[-1][2][f_dst[0]])
                        # print("----")
                        final_path_info = add_merge_info(G_merge, src, final_path_info, group_id, -1, quality_list)
                        
                        if final_path_info[group_id] == [[], [], [], []]:
                            final_path_info.insert(group_id+1, final_path_info[-1])
                            final_path_info.pop(group_id)
                        final_path_info.pop(-1)
                    
    # print(failed_group)
    # print(not_failed_dsts)
    for i in not_failed_dsts:
        failed_group.remove(i)
    # print(failed_group)

    # Merge Path 
    for i in range(len(final_path_info)):
        merge_path_ans = merge_path(G, G_final, src, quality_list, final_path_info, i, weight)
        G_final = copy.deepcopy(merge_path_ans[0])
        final_path_info = copy.deepcopy(merge_path_ans[1])        

    # print(final_path_info)
    # print(Graph.cal_total_cost(G_final, weight, True))

    count_vnf = 0
    delay = [0]
    count_group = 0
    for i in range(len(final_path_info)):
        count_vnf += Graph.count_vnf(final_path_info[i][1])
        delay.append(Graph.max_len(final_path_info[i][0]))
        count_group += 1

    # count_n = 0
    # for n,dic in G_final.nodes(data=True):
    #     if dic['com_capacity'] < 0: count_n+=1
    #     if dic['mem_capacity'] < 0: count_n+=1

    # for n1,n2,dic in G_final.edges(data=True):
    #     if dic['bandwidth'] < 0: count_n+=1

    # print("neg=",count_n)

    return (G_final, failed_group, count_vnf, max(delay))

# Merge path from src to dst for one group.
# group_info[i] = [path_set, sfc, data_rate, dsts] 
def merge_path(G, G_min, src, quality_list, group_info, index, weight):
    G_merge = copy.deepcopy(G_min)

    video_type = [q for q in quality_list]
    video_type.reverse()

    ### Use list to record the saved cost of merging two path i,j.
    ### And select the maximum saved cost to merge path.
    ### (Reduce question complexity with setting shortest path length threshold)
    new_group_info = copy.deepcopy(group_info)
    path_set = new_group_info[index][0]
    sfc = new_group_info[index][1]
    data_rate = new_group_info[index][2]
    this_group_dsts = new_group_info[index][3]
    path_length_threshold = 3 #len(G_min.edges)

    dst_list = [d[0] for d in this_group_dsts]

    if path_set == []: 
        return (G_merge,new_group_info)
    
    # print('------ merge path begin ------')
    # print(path_set)
    # print(sfc)
    # # print(data_rate)
    # print(this_group_dsts)
    # print('-------------')
    # print("nor. cost=",Graph.cal_total_cost_normalize(G, G_merge, weight, True))
    # print("cost=",Graph.cal_total_cost(G_merge, weight, True))

    dsts_without_vnf = []

    # Count the number of sharing VNF on node v
    count_VNF = {}
    for d in sfc:
        if len(sfc[d]) == 0: 
            dsts_without_vnf.append(d)
            continue
        place_node = sfc[d][0][1]
        if place_node not in count_VNF:
            count_VNF[place_node] = [1,[d]]
        else:
            count_VNF[place_node][0] += 1
            count_VNF[place_node][1].append(d)

    # print("count_VNF:",count_VNF)
    pre_time = time.time()

    is_finish = False
    while is_finish == False:
        cost_metric = dict()
        # orig_cost = Graph.cal_total_cost(G_merge, weight, True)
        orig_cost = Graph.cal_total_cost_normalize(G, G_merge, weight, True)
        # print("orig cost = ", orig_cost)
        for i in range(len(dsts_without_vnf)):
            dst1 = dsts_without_vnf[i]
            for j in range(i+1, len(dsts_without_vnf)):
                dst2 = dsts_without_vnf[j]
                # print('=================', dst1, ", ", dst2)
               
                shortest_path_between_dsts = nx.algorithms.shortest_paths.dijkstra_path(G_merge, dst1, dst2, weight='weight')
                
                if len(shortest_path_between_dsts) - 2 > path_length_threshold: continue

                merge_cost = orig_cost[0]
                merge_group_info = -1
                merge_G = nx.Graph()

                for m in shortest_path_between_dsts:
                    if m == src:
                        continue

                    merge_info = [src, dst1, dst2, m]

                    tmp_group_info = copy.deepcopy(new_group_info)
                    tmp_group_info[index] = update_merge_info(G_merge, merge_info, -1, new_group_info[index])

                    G_tmp = copy.deepcopy(G)
                    for k in range(len(tmp_group_info)):
                        G_tmp = update_graph(G_tmp, tmp_group_info[k])
                    # total_cost = Graph.cal_total_cost(G_tmp, weight, True)
                    total_cost = Graph.cal_total_cost_normalize(G, G_tmp, weight, True)
                    # print("merge cost = ", total_cost)

                    if total_cost[0] < merge_cost:
                        merge_cost = total_cost[0]
                        merge_group_info = copy.deepcopy(tmp_group_info)
                        merge_G = copy.deepcopy(G_tmp)
    
                cost_metric[(dst1,dst2)] = (round(orig_cost[0] - merge_cost, 2), merge_group_info, merge_G)
                
        # Choose the maximum save_cost (orig_cost - merge_cost) and merge the paths with merge_node.
        if cost_metric == {}:
            is_finish = True
            # print("no merge")
            break
        merge_result = max(cost_metric, key=lambda k: cost_metric[k][0])
        
        if cost_metric[merge_result][0] <= 0:
            is_finish = True
            # print("no merge")
            break

        new_group_info = copy.deepcopy(cost_metric[merge_result][1])
        G_merge = copy.deepcopy(cost_metric[merge_result][2])

    # running_time_main = time.time() - pre_time
    # print("merge no VNF =",time.time() - pre_time)
    # print("merge cost=",Graph.cal_total_cost(G_merge, weight, True))
    # print("dsts_without_vnf=",Graph.cal_total_cost_normalize(G, G_merge, weight, True))

    count_VNF_list = list(count_VNF.keys())

    # 把其他部署在比較少共用節點的VNF移動位置
    # 想想看怎麼樣能移動整坨，增加VNF嗎，閾值為多少要增加新的

    pre_time = time.time()
    # Find migrate VNF node
    for node in count_VNF:
        path_set = new_group_info[index][0]
        sfc = new_group_info[index][1]

        same_VNF_dsts = count_VNF[node][1]

        common_len = Graph.find_all_common_path_len(dst_list, same_VNF_dsts, path_set)
        one_dst_in_count_VNF = same_VNF_dsts[0]
        if common_len > sfc[one_dst_in_count_VNF][0][2]:
            migrate_node = path_set[one_dst_in_count_VNF][common_len-1]
            count_VNF_list.append(migrate_node)

    # print("720=",time.time()-pre_time)

    pre_time = time.time()
    count_VNF = {k: v for k, v in sorted(count_VNF.items(), key=lambda item: item[1])}
    # print(count_VNF)

    # Merge Path or Migrate VNF
    for node in count_VNF:
        same_VNF_dsts = count_VNF[node][1]

        for d in same_VNF_dsts:
            # orig_cost = Graph.cal_total_cost(G_merge, weight, True)
            orig_cost = Graph.cal_total_cost_normalize(G, G_merge, weight, True)
            merge_cost = orig_cost[0]
            merge_group_info = new_group_info
            merge_G = copy.deepcopy(G_merge)
            
            for tmp_node in count_VNF_list:
                if tmp_node == d or tmp_node == node: continue

                merge_info = [src, d, d, tmp_node]

                tmp_group_info = copy.deepcopy(new_group_info)
                tmp_group_info[index] = update_merge_info(G_merge, merge_info, 0, new_group_info[index])

                G_tmp = copy.deepcopy(G)
                for k in range(len(tmp_group_info)):
                    G_tmp = update_graph(G_tmp, tmp_group_info[k])

                # tmp_cost = Graph.cal_total_cost(G_tmp, weight, True)
                tmp_cost = Graph.cal_total_cost_normalize(G, G_tmp, weight, True)
                
                if tmp_cost[0] < merge_cost:
                    merge_cost = tmp_cost[0]
                    merge_group_info = copy.deepcopy(tmp_group_info)
                    merge_G = copy.deepcopy(G_tmp)
            
            new_group_info = copy.deepcopy(merge_group_info)
            G_merge = copy.deepcopy(merge_G)
            # print("same_VNF_dsts=",Graph.cal_total_cost_normalize(G, G_merge, weight, True))

    # print("759=",time.time()-pre_time)

    final_group_info = copy.deepcopy(new_group_info)

    G_merge = copy.deepcopy(G)
    for k in range(len(final_group_info)):
        G_merge = update_graph(G_merge, final_group_info[k])
        
    # path_set = new_group_info[index][0]
    # sfc = new_group_info[index][1]
    # data_rate = new_group_info[index][2]

    # print('------------')
    # print(final_group_info[index][0])
    # print(final_group_info[index][1])
    # print(final_group_info[index][2])
    # # print("merge path count_m =", count_m)
    # print("final cost=",Graph.cal_total_cost(G_merge, weight, True))
    # print("final=",Graph.cal_total_cost_normalize(G, G_merge, weight, True))
    # print('-------------')

    return (G_merge,final_group_info)

# Update path_set, data_rate and sfc after merging for one group.
# merge_info = [src, dst1, dst2, place_node]
def update_merge_info(G_min, merge_info, vnf_id, group_info):
    path_set = group_info[0]
    sfc = group_info[1]
    all_data_rate = group_info[2]

    src = merge_info[0]
    dst1 = merge_info[1]
    dst2 = merge_info[2]
    place_node = merge_info[3]
    
    shortest_path_before_dst = nx.algorithms.shortest_paths.dijkstra_path(G_min, src, place_node, weight='weight')
    shortest_path_after_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, dst1, weight='weight')[1:]
    shortest_path_after_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, dst2, weight='weight')[1:]

    # print("------- update_group_info -------")
    # print("place_node = ", place_node)
    # print('======= dst1 ', dst1_info[0],'========')
    # print(path_set[dst1_info[0]], sfc[dst1_info[0]])
    # print(shortest_path_before_dst1)
    # print(shortest_path_after_dst1)
    # print('======= dst2 ', dst2_info[0],'========')
    # print(path_set[dst2_info[0]], sfc[dst2_info[0]])
    # print(shortest_path_before_dst2)
    # print(shortest_path_after_dst2)
    # print()

    ### path_set
    new_path_set = copy.deepcopy(path_set)
    
    new_path_set[dst1] = shortest_path_before_dst + shortest_path_after_dst1
    new_path_set[dst2] = shortest_path_before_dst + shortest_path_after_dst2

    ### data_rate
    new_data_rate = update_data_rate(dst1, merge_info, shortest_path_before_dst, shortest_path_after_dst1, all_data_rate)
    new_data_rate = update_data_rate(dst2, merge_info, shortest_path_before_dst, shortest_path_after_dst2, new_data_rate)
    
    ### sfc
    new_sfc = copy.deepcopy(sfc)
    if vnf_id == 0:
        place_node_id_dst1 = len(shortest_path_before_dst) - 1
        place_node_id_dst2 = len(shortest_path_before_dst) - 1

        new_sfc[dst1][vnf_id] = [sfc[dst1][vnf_id][0], place_node, place_node_id_dst1]
        new_sfc[dst2][vnf_id] = [sfc[dst2][vnf_id][0], place_node, place_node_id_dst2]

    
    # print('------------')
    # print(new_path_set[dst1_info[0]], new_path_set[dst2_info[0]])
    # print(new_sfc[dst1_info[0]], new_sfc[dst2_info[0]])
    # print(new_data_rate[dst1_info[0]], "\n", new_data_rate[dst2_info[0]])
    # print('-------------------------------------!!')

    return [new_path_set, new_sfc, new_data_rate, group_info[3]]

# Update data_rate after merging.
# merge_info = [src, dst1, dst2, place_node] 
def update_data_rate(dst, merge_info, shortest_path_before, shortest_path_after, data_rate):
    new_data_rate = copy.deepcopy(data_rate)

    # print(sfc[dst][vnf_id][2], data_rate[dst], vnf_id)

    input_q = data_rate[dst][0][1]
    output_q = data_rate[dst][-1][1]

    input_data = data_rate[dst][0][2]
    output_data = data_rate[dst][-1][2]
    # print(dst)
    
    new_data_rate[dst] = []

    # print("input: ", input_data, input_q, ", output: ", output_data, output_q)
    # print(vnf_id, sfc[dst])
    # print(shortest_path_before)
    # print(shortest_path_after)

    for n in shortest_path_before:
        if n == shortest_path_before[-1]: break
        new_data_rate[dst].append((n, input_q, input_data))

    if merge_info[3] != dst:
        new_data_rate[dst].append((merge_info[3], output_q, output_data))
    
    for n in shortest_path_after:
        if n == dst: break
        new_data_rate[dst].append((n, output_q, output_data))
    
    # print(new_data_rate[dst])

    return new_data_rate

# Update sfc
def update_sfc(dst, sfc):
    new_sfc = copy.deepcopy(sfc)

    input_q = sfc[dst][0][0][-2]
    output_q = sfc[dst][0][0][-1]
    if len(sfc[dst]) > 1:
        new_sfc[dst] = []
    for v_id in range(1,len(sfc[dst])):
        if sfc[dst][v_id-1][2] == sfc[dst][v_id][2]:
            output_q = sfc[dst][v_id][0][-1]
        else:
            vnf = "t_" + str(input_q) + str(output_q)
            new_sfc[dst].append([vnf, sfc[dst][v_id-1][1], sfc[dst][v_id-1][2]])
            input_q = sfc[dst][v_id][0][-2]
            output_q = sfc[dst][v_id][0][-1]

        if v_id == len(sfc[dst])-1:
            vnf = "t_" + str(input_q) + str(output_q)
            new_sfc[dst].append([vnf, sfc[dst][v_id][1], sfc[dst][v_id][2]])

    return new_sfc

# Add additional VNF to path
def add_vnf_to_path(G, path_info, src, send_data, quality_list):
    path_set = path_info[0]
    sfc = path_info[1]
    data_rate = path_info[2]

    new_path_set = copy.deepcopy(path_set)
    new_sfc = copy.deepcopy(sfc)
    new_data_rate = copy.deepcopy(data_rate)

    for dst in path_set:
        if len(new_sfc[dst]) == 0:
            if len(path_set[dst]) >= 3:
                new_sfc[dst].append(['t', path_set[dst][1], 1])
                new_data_rate[dst][0] = (src, send_data, quality_list[send_data])
            else:
                # change path_set, data_rate
                find_distance = 0 # the length range of finding node away from last_node that placing VNF
                place_flag = False # whether placing VNF or not (share or initiate), 0: not place, 1: place, 2: the node can't place
        
                while place_flag == False:
                    find_distance += 1
                    
                    alternate_nodes_len = nx.single_source_shortest_path_length(G, src, find_distance)
                    alternate_nodes = list(n for n in alternate_nodes_len)

                    for i in alternate_nodes:
                        if i == src or i == dst: continue
                        
                        # If find_distance > 1, the node that distance = 1 have been searched
                        # it don't have to consider the node again
                        if find_distance > 1 and alternate_nodes_len[i] <= find_distance-1:
                            continue

                        new_sfc[dst].append(['t', i, 1])

                        shortest_path_before = nx.algorithms.shortest_paths.dijkstra_path(G, src, i, weight='weight')
                        shortest_path_after = nx.algorithms.shortest_paths.dijkstra_path(G, i, dst, weight='weight')[1:]
                        
                        new_path_set[dst] = shortest_path_before + shortest_path_after
                        
                        dst_info = [src, dst, -1, i] 
                        new_data_rate[dst].insert(0, (src, send_data, quality_list[send_data]))
                        new_data_rate = update_data_rate(dst, dst_info, shortest_path_before, shortest_path_after, new_data_rate)

                        # print(new_data_rate[dst])
                        place_flag = True
                        break
                continue
        else:
            vnf_index = new_sfc[dst][0][2]
            for i in range(vnf_index):
                new_data_rate[dst][i] = (new_path_set[dst][i], send_data, quality_list[send_data])

            output_data = exp_set.cal_transcode_bitrate(quality_list[send_data], send_data, new_data_rate[dst][vnf_index][1])
            for i in range(vnf_index, len(new_data_rate[dst])):
                new_data_rate[dst][i] = (new_path_set[dst][i], new_data_rate[dst][i][1], output_data)

    # print('------ new ------')
    # print(new_path_set)
    # print(new_sfc)
    # print(new_data_rate)

    return [new_path_set, new_sfc, new_data_rate]

# update the merge group information, 
# if their send data are not same
def add_merge_info(G, src, path_info, first_group, second_group, quality_list):
    send_data_1 = path_info[first_group][2][next(iter(path_info[first_group][2]))][0][1]
    send_data_2 = path_info[second_group][2][next(iter(path_info[second_group][2]))][0][1]

    merge_group_info = copy.deepcopy(path_info)

    if send_data_1 == send_data_2:
        for i in range(3):
            merge_group_info[first_group][i].update(path_info[second_group][i])
        merge_group_info[first_group][3] += path_info[second_group][3]
        merge_group_info[second_group] = [[],[],[],[]]
    elif quality_list[send_data_1] > quality_list[send_data_2]:
        # Add VNF to dst2
        tmp_path_info = add_vnf_to_path(G, merge_group_info[second_group], src, send_data_1, quality_list)
        for i in range(3):
            merge_group_info[first_group][i].update(tmp_path_info[i])
        merge_group_info[first_group][3] += path_info[second_group][3]
        merge_group_info[second_group] = [[],[],[],[]]
    else:
        # Add VNF to dst1
        tmp_path_info = add_vnf_to_path(G, merge_group_info[first_group], src, send_data_2, quality_list)
        for i in range(3):
            merge_group_info[second_group][i].update(tmp_path_info[i])
        merge_group_info[second_group][3] += path_info[first_group][3]
        merge_group_info[first_group] = [[],[],[],[]]

    return merge_group_info

# Use new_path_info to create new path on G for one group.
def update_graph(G, new_path_info):
    path_set = new_path_info[0]
    sfc = new_path_info[1]
    data_rate = new_path_info[2]
    sort_dsts = new_path_info[3]

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
                vnf_find = "t"

                # There is vnf instance on node j
                if vnf_find not in list(v[0][0] for v in G_min.nodes[j]['vnf']):
                    G_min.nodes[j]['mem_capacity'] -= 1
                is_process_data = Graph.add_new_processing_data_main(G_min, multicast_path, tmp_list, dst, i, vnf, data_rate[dst][i-1], data_rate)
                sfc_index += 1

            # connect edge
            e = (path_set[dst][i-1], path_set[dst][i])
            Graph.add_new_edge(G_min, multicast_path, tmp_list, dst, e, data_rate[dst][i-1], data_rate)

    return G_min

# 1. （目前）一條一條路徑做合併
# 問題：計算成本時，無法精準計算（邊、點有無重用）
#      如何決定誰要跟誰合併？如何制定結束合併的條件？

# 2. 在全部組走完後，重建一個加權圖（計算哪個邊、點經過次數較多），以重新搜尋最短路徑盡可能共享
# 問題：如何設定權重，以避免路徑壅塞？
#      這樣做的話，分組有意義嗎？

# 3. 在全部組走完後，以最後一個有部署vnf的點為new_dst，想辦法合併前半部的路（將問題空間降低）
# 問題：如何合併前半部的路，將所有可能列出來，去選擇成本較低者嗎？