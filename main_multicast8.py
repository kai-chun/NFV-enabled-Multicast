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

        # remove_edges = list((n1,n2) for n1,n2,dic in G_min.edges(data=True) if dic['bandwidth'] < data_rate[dst][-1][2])
        # G_tmp.remove_edges_from(remove_edges)

        # try:
        #     shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_tmp, src, dst, weight='weight')
        # except:
        #     failed_dsts.append(d)
        #     pass

        # if d in failed_dsts: continue

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
            if G_min.edges[e]['bandwidth'] < data_rate[dst][-1][2]:
                is_connect = False
                break
            Graph.add_new_edge(G_min, multicast_path_min, shortest_path_set, dst, e, data_rate[dst][-1], data_rate)

        if is_connect == False:
            G_min = copy.deepcopy(G_tmp)
            multicast_path_min = copy.deepcopy(tmp_path)
            shortest_path_set = copy.deepcopy(tmp_path_set)
            data_rate = copy.deepcopy(tmp_data_rate)
            index_sfc = copy.deepcopy(tmp_index_sfc)
            failed_dsts.append(d)

    #
    # A corrective subroutine that places the missing NF instances 
    # on the closest NFV node from the multicast topology.
    #
    for d in sort_dsts:
        if d[0] in list(f[0] for f in failed_dsts):
            index_sfc.pop(d[0],None)
    # print(index_sfc)
    # print(failed_dsts)
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
                    failed_dsts = copy.deepcopy(sort_dsts)
                    return (G, nx.Graph(), list(), dict(), dict(), failed_dsts, failed_dsts, sfc)
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
                        if G_min.edges[e]['bandwidth'] < data_rate[dst][-2][2]:
                            is_connect = False
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                    else:
                        for j in range(len(shortest_path)-1):
                            if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                update_shortest_path_set[dst].append(shortest_path[j+1])
                            if shortest_path[j] not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            if j != 0:
                                data_rate[dst].insert(-1, (shortest_path[j],data_rate[dst][0][1],data_rate[dst][0][2]))
                                #data_rate[dst].append((shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            if G_min.edges[e]['bandwidth'] < data_rate[dst][-2][2]:
                                is_connect = False
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)

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
                        # If previous_node to i don't have path then build it
                        if G_min.edges[e]['bandwidth'] < data_rate[dst][-2][2]:
                            is_connect = False
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)
                    else:   
                        for j in range(len(shortest_path)-1):
                            if update_shortest_path_set[dst][-1] != shortest_path[j+1]:
                                update_shortest_path_set[dst].append(shortest_path[j+1])
                            if shortest_path[j] not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            if j != 0:
                                data_rate[dst].insert(-1, (shortest_path[j],data_rate[dst][0][1],data_rate[dst][0][2]))
                                #data_rate[dst].append((shortest_path[j],data_rate[dst][-1][1],data_rate[dst][-1][2]))
                            if G_min.edges[e]['bandwidth'] < data_rate[dst][-2][2]:
                                is_connect = False
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][-2], data_rate)

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

        #shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, dst, weight='weight')
        
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
    ### print data
    dst_num = 0
    cost_record = [[],[]]

    failed_group = list()
    all_group = list()
    for g in all_group_result:
        if nx.classes.function.is_empty(g[1]):
            failed_group.extend(g[6])
            dst_num += len(g[6])
        else:
            all_group.append([g[2],g[3],g[4],g[5]])
            dst_num += len(g[5])
            if len(g[6]) != 0:
                failed_group.extend(g[6])
                dst_num += len(g[6])

    if len(failed_group) != 0:
        group_info = dict()
        sfc = dict()
        for g in failed_group:
            group_info[g[0]] = g[1]
            sfc[g[0]] = []

        video_type = [q for q in quality_list]
        best_quality = video_type[max(video_type.index(group_info[g]) for g in group_info)]
        service_fail = (src, group_info, sfc, best_quality)
        re_route_fail_result = search_multipath(G_min, service_fail, quality_list)
        if not nx.classes.function.is_empty(re_route_fail_result[1]):
            print("more")
            all_group.append([re_route_fail_result[2],re_route_fail_result[3],re_route_fail_result[4],re_route_fail_result[5]])
        failed_group = re_route_fail_result[6]
    
    sort_group = copy.deepcopy(all_group)#sorted(all_group, key=lambda g: Graph.count_instance(g[0]), reverse=True)

    new_path_info = []

    # print("failed_group: ",failed_group)
    # print("group num = ", len(sort_group))
    # print('---------------------')
    # print("original path: ", Graph.cal_total_cost(G_min, weight, True))
    # # print(G_min.edges(data=True))
    # print('---------------------')
    
    if len(sort_group) == 0:
        new_path_info.append(merge_path(G, G_min, src, quality_list, sort_group, 0, weight)[0])
    else:
        for s in sort_group:
            new_path_info.append(s)

    # print(sort_group[0])
    
    G_merge = copy.deepcopy(G)
    for i in range(len(new_path_info)):
        G_merge = update_graph(G_merge, new_path_info[i])

    merge_group_info = copy.deepcopy(new_path_info)
    # print(merge_group_info)
    cost_record[0].append("original")
    cost_record[1].append(Graph.cal_total_cost(G_merge, weight, True)[0])

    # print('---------------------')
    # print("merge path: ", Graph.cal_total_cost(G_merge, weight, True))
    # # print(G_new.edges(data=True))
    # print('---------------------')
    
    count_m = 0

    # Calculate the cost of merge 2 groups, and choose the minimum cost merging strategy.
    is_finish = False
    merge_weight = (0.6, 0.4, 1)
    while is_finish == False:
        cost_metric = dict()

        # Calculate original cost
        orig_cost_groups = Graph.cal_total_cost(G_merge, merge_weight, True)
        for i in range(len(sort_group)):
            for j in range(i+1, len(sort_group)):
                if len(merge_group_info[i][3]) == 0 or len(merge_group_info[j][3]) == 0:
                    cost_metric[(i,j)] = [0,0]
                    continue

                # print('------',i,'-------')
                # print(merge_group_info[i])
                # print('------',j,'-------')
                # print(merge_group_info[j])

                # Calculate merging cost
                tmp_merge_info = add_merge_info(G_merge, src, merge_group_info, i, j, quality_list)
                G_tmp = copy.deepcopy(G)

                for m in range(len(tmp_merge_info)):
                    G_tmp = update_graph(G_tmp, tmp_merge_info[m])

                tmp_cost = Graph.cal_total_cost(G_tmp, merge_weight, True)[0]
                # print("merge group =",Graph.cal_total_cost(G_tmp, weight, True))
                
                new_path_info_merge = merge_path(G, G_tmp, src, quality_list, tmp_merge_info, i, merge_weight)
                
                G_tmp = copy.deepcopy(G)
                for m in range(len(new_path_info_merge)):
                    G_tmp = update_graph(G_tmp, new_path_info_merge[m])
                
                merge_cost_groups = Graph.cal_total_cost(G_tmp, merge_weight, True)

                cost_metric[(i,j)] = [round(orig_cost_groups[0] - merge_cost_groups[0], 2), new_path_info_merge, tmp_cost, G_tmp]

                # print("merge group and path = ", merge_cost_groups)
        
        if cost_metric == {}:
            is_finish = True
            # print("no merge")
            break
        
        merge_result = max(cost_metric, key=lambda k: cost_metric[k][0])
        if cost_metric[merge_result][0] <= 0:
            is_finish = True
            # print("no merge")
            break
        
        # print(merge_result, cost_metric[merge_result])
        # print('---')

        cost_record[0].append("merge_group")
        cost_record[1].append(cost_metric[merge_result][2])

        count_m += 1
        merge_group_info = copy.deepcopy(cost_metric[merge_result][1])

        G_merge = copy.deepcopy(cost_metric[merge_result][3])

        cost_record[0].append("merge_path")
        cost_record[1].append(Graph.cal_total_cost(G_merge, weight, True)[0])
        
        # print("merge group and path = ", merge_cost_groups)
    
    count_vnf = 0
    delay = [0]
    count_group = 0
    for i in range(len(merge_group_info)):
        count_vnf += Graph.count_vnf(merge_group_info[i][1])
        delay.append(Graph.max_len(merge_group_info[i][0]))
        count_group += 1
    # print(merge_group_info)
    # print("merge group count_m = ",count_m)
    # print("merge group: ", Graph.cal_total_cost(G_merge, weight, True))
    
    ### print data
    # file_path = 'exp_data/merge_iter/a8/'+str(len(G.nodes()))+'_d'+str(dst_num)+'_bw20_g1.xlsx'
    # data = pd.DataFrame({"Merge":cost_record[0],"Cost":cost_record[1]})
    # data.to_excel(file_path, index=False)
    return (G_merge, failed_group, count_vnf, max(delay))

# Merge path from src to dst for one group.
def merge_path(G, G_min, src, quality_list, group_info, index, weight):
    G_merge = copy.deepcopy(G_min)

    video_type = [q for q in quality_list]
    video_type.reverse()

    ### 用一個矩陣記錄兩兩路徑間合併某節點後節省的成本，選最低者
    ### 過程中用 threshold（hop數）降低複雜度
    new_group_info = copy.deepcopy(group_info)
    path_set = new_group_info[index][0]
    sfc = new_group_info[index][1]
    data_rate = new_group_info[index][2]
    this_group_dsts = new_group_info[index][3]
    path_length_threshold = 5 #len(G_min.edges)

    if path_set == []: 
        return new_group_info
    
    # print('------ merge path begin ------')
    # print(path_set)
    # print(sfc)
    # print(data_rate)
    # print('-------------')
    
    ### Merge with paths of other dsts.
    vnf_id = 0
    count_m = 0

    is_finish = False
    while is_finish == False:
        sfc = new_group_info[index][1]
        
        cost_metric = dict()
        orig_cost = Graph.cal_total_cost(G_merge, weight, True)
        # print("orig cost = ", orig_cost)
        for i in range(len(this_group_dsts)):
            dst1 = this_group_dsts[i][0]
            for j in range(i+1, len(this_group_dsts)):
                dst2 = this_group_dsts[j][0]
                # print('=================', dst1, ", ", dst2)

                if vnf_id < len(sfc[dst1]) and vnf_id < len(sfc[dst2]):
                    shortest_path_between_vnfs = nx.algorithms.shortest_paths.dijkstra_path(G_merge, sfc[dst1][vnf_id][1], sfc[dst2][vnf_id][1], weight='weight')
                    if len(shortest_path_between_vnfs) - 2 > path_length_threshold:
                        cost_metric[(dst1,dst2)] = (0, -1, [], [])
                    else:
                        last_node_id_dst1 = sfc[dst1][vnf_id-1][2]
                        last_node_id_dst2 = sfc[dst2][vnf_id-1][2]

                        if vnf_id == 0: # then last_node is src
                            last_node_id_dst1 = 0 
                            last_node_id_dst2 = 0

                        merge_cost = orig_cost[0]
                        merge_node = -1
                        merge_G = nx.Graph()

                        if vnf_id == len(sfc[dst1])-1:
                            next_node_id_dst1 = -1
                        else:
                            next_node_id_dst1 = sfc[dst1][vnf_id][2]
                        
                        if vnf_id == len(sfc[dst2])-1:
                            next_node_id_dst2 = -1
                        else:
                            next_node_id_dst2 = sfc[dst2][vnf_id][2]

                        # Iterative all node between v[1] and sfc[main_dst][j][1],
                        # calculate their cost, and get the node has minimim cost to merge two path.
                        for m in shortest_path_between_vnfs:
                            if m == src or m == dst1 or m == dst2 or (sfc[dst1][vnf_id][0][0] != sfc[dst2][vnf_id][0][0]):
                                continue

                            dst1_info = [dst1, last_node_id_dst1, m, next_node_id_dst1]
                            dst2_info = [dst2, last_node_id_dst2, m, next_node_id_dst2]

                            vnf_find = sfc[dst1][vnf_id][0]
                            if vnf_find[0] == 't':
                                vnf_find = "t"

                            tmp_group_info = copy.deepcopy(new_group_info)
                            tmp_group_info[index] = merge_group_info(G_merge, dst1_info, dst2_info, vnf_id, new_group_info[index])

                            G_tmp = copy.deepcopy(G)
                            for k in range(len(tmp_group_info)):
                                G_tmp = update_graph(G_tmp, tmp_group_info[k])
                            total_cost = Graph.cal_total_cost(G_tmp, weight, True)
                            # print("merge cost = ", total_cost)

                            if total_cost[0] < merge_cost:
                                merge_cost = total_cost[0]
                                merge_node = m
                                merge_G = copy.deepcopy(G_tmp)
            
                        dst1_info = [dst1, last_node_id_dst1, merge_node, next_node_id_dst1]
                        dst2_info = [dst2, last_node_id_dst2, merge_node, next_node_id_dst2]
                        cost_metric[(dst1,dst2)] = (round(orig_cost[0] - merge_cost, 2), merge_node, dst1_info, dst2_info, tmp_group_info, merge_G)
        
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
        # print("cost_metric: ", cost_metric)
        # print(merge_result, cost_metric[merge_result])
        
        # print('------------')
        # print(new_group_info[index][0][merge_result[0]], new_group_info[index][0][merge_result[1]])
        # print(new_group_info[index][1][merge_result[0]], new_group_info[index][1][merge_result[1]])
        # print(new_group_info[index][2][merge_result[0]], new_group_info[index][2][merge_result[1]])
        # print('-----------')

        dst1_info = cost_metric[merge_result][2]
        dst2_info = cost_metric[merge_result][3]

        new_group_info = copy.deepcopy(cost_metric[merge_result][4])
        G_merge = copy.deepcopy(cost_metric[merge_result][5])
        # new_group_info[index] = merge_group_info(G_merge, dst1_info, dst2_info, vnf_id, new_group_info[index])
        # G_merge = copy.deepcopy(G)
        # for k in range(len(new_group_info)):
        #     G_merge = update_graph(G_merge, new_group_info[k])
        
        count_m += 1
        
    # path_set = new_group_info[index][0]
    # sfc = new_group_info[index][1]
    # data_rate = new_group_info[index][2]

    # print('------------')
    # print(path_set)
    # print(sfc)
    # print(data_rate)
    print("merge path count_m =", count_m)
    # print(Graph.cal_total_cost(G_merge, weight, True))
    # print('-------------')

    return new_group_info

# Calculate total cost from src_vnf to dst_vnf.
# dst_info = [dst, last_node_id_dst, place_node, next_node_id_dst]
def cal_cost_merge(G_min, dst1_info, dst2_info, vnf_id, group_info, video_type, weight):
    path_set = group_info[0]
    sfc = group_info[1]
    all_data_rate = group_info[2]

    place_node = dst1_info[2]

    # print(dst1_info[2])
    # print('--- ', dst1_info[0], ' ---')
    # print(path_set[dst1_info[0]])
    # print(all_data_rate[dst1_info[0]])
    # print('--- ', dst2_info[0], ' ---')
    # print(path_set[dst2_info[0]])
    # print(all_data_rate[dst2_info[0]])
    
    # Transimission cost - dst1, before place_node
    trans_cost = 0

    tmp_path_set = copy.deepcopy(path_set)
    
    shortest_path_before_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst1_info[0]][dst1_info[1]], place_node, weight='weight')
    shortest_path_after_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst1_info[0]][dst1_info[3]], weight='weight')[1:]
    shortest_path_before_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst2_info[0]][dst2_info[1]], place_node, weight='weight')
    shortest_path_after_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst2_info[0]][dst2_info[3]], weight='weight')[1:]

    tmp_path_set[dst1_info[0]][dst1_info[1]:] = shortest_path_before_dst1
    tmp_path_set[dst2_info[0]][dst2_info[1]:] = []
    tmp_data_rate = update_data_rate(dst1_info, shortest_path_before_dst1, shortest_path_after_dst1, vnf_id, all_data_rate, sfc)
    tmp_data_rate = update_data_rate(dst2_info, shortest_path_before_dst2, shortest_path_after_dst2, vnf_id, tmp_data_rate, sfc)

    place_node_id_dst1 = len(tmp_path_set[dst1_info[0]]) - 1
    commom_len_dst1 = Graph.find_common_path_len_edge(dst1_info[0], tmp_path_set, tmp_data_rate) 
    if place_node_id_dst1 >= commom_len_dst1: # If not multicast link
        max_len = max(commom_len_dst1-1, dst1_info[1])
        trans_cost += all_data_rate[dst1_info[0]][dst1_info[1]][2] * (place_node_id_dst1 - max_len)

    # Check bandwidth resoure enough or not
    for i in range(commom_len_dst1-1, place_node_id_dst1-1):
        e = (tmp_path_set[dst1_info[0]][i], tmp_path_set[dst1_info[0]][i+1])
        if G_min.edges[e]['bandwidth'] < all_data_rate[dst1_info[0]][dst1_info[1]][2]:
            trans_cost += float('inf')
    
    # Transimission cost - dst2, before place_node
    tmp_path_set[dst2_info[0]][dst2_info[1]:] = shortest_path_before_dst2
    
    place_node_id_dst2 = len(tmp_path_set[dst2_info[0]]) - 1
    commom_len_dst2 = Graph.find_common_path_len_edge(dst2_info[0], tmp_path_set, tmp_data_rate) 
    if place_node_id_dst2 >= commom_len_dst2: # If not multicast link
        max_len = max(commom_len_dst2-1, dst2_info[1])
        trans_cost += all_data_rate[dst2_info[0]][dst2_info[1]][2] * (place_node_id_dst2 - max_len)

    # Check bandwidth resoure enough or not
    for i in range(commom_len_dst2-1, place_node_id_dst2-1):
        e = (tmp_path_set[dst2_info[0]][i], tmp_path_set[dst2_info[0]][i+1])
        if G_min.edges[e]['bandwidth'] < all_data_rate[dst2_info[0]][dst2_info[1]][2]:
            trans_cost += float('inf')

    # Transimission cost - dst1, after place_node
    tmp_path_set[dst1_info[0]].extend(shortest_path_after_dst1)
    commom_len_dst1 = Graph.find_common_path_len_edge(dst1_info[0], tmp_path_set, tmp_data_rate) 
    if len(tmp_path_set[dst1_info[0]]) - 1 >= commom_len_dst1: # If not multicast link
        max_len = max(commom_len_dst1-1, place_node_id_dst1)
        trans_cost += all_data_rate[dst1_info[0]][sfc[dst1_info[0]][vnf_id][2]][2] * (len(tmp_path_set[dst1_info[0]]) - 1 - max_len)

    # Check bandwidth resoure enough or not
    for i in range(commom_len_dst1-1, len(tmp_path_set[dst1_info[0]])-2):
        e = (tmp_path_set[dst1_info[0]][i], tmp_path_set[dst1_info[0]][i+1])
        if G_min.edges[e]['bandwidth'] < all_data_rate[dst1_info[0]][sfc[dst1_info[0]][vnf_id][2]][2]:
            trans_cost += float('inf')

    # Transimission cost - dst2, after place_node
    tmp_path_set[dst2_info[0]].extend(shortest_path_after_dst2)
    commom_len_dst2 = Graph.find_common_path_len_edge(dst2_info[0], tmp_path_set, tmp_data_rate) 
    if len(tmp_path_set[dst2_info[0]]) - 1 >= commom_len_dst2: # If not multicast link
        max_len = max(commom_len_dst2-1, place_node_id_dst2)
        trans_cost += all_data_rate[dst2_info[0]][sfc[dst2_info[0]][vnf_id][2]][2] * (len(tmp_path_set[dst2_info[0]]) - 1 - max_len)

    # Check bandwidth resoure enough or not
    for i in range(commom_len_dst2-1, len(tmp_path_set[dst2_info[0]])-2):
        e = (tmp_path_set[dst2_info[0]][i], tmp_path_set[dst2_info[0]][i+1])
        if G_min.edges[e]['bandwidth'] < all_data_rate[dst2_info[0]][sfc[dst2_info[0]][vnf_id][2]][2]:
            trans_cost += float('inf')

    # Placing cost
    if G_min.nodes[place_node]['mem_capacity'] > 0:
        place_cost = 1
    else:
        place_cost = float('inf')
    for d in sfc:
        if place_cost == 0: break
        for v in sfc[d]:
            if d != dst1_info[0] and d != dst2_info[0] and v[1] == place_node:
                place_cost = 0
                break
            if (d == dst1_info[0] or d == dst2_info[0]) and v[1] == place_node and v[2] != place_node_id_dst1 and v[2] != place_node_id_dst2:
                place_cost = 0
                break
    
    # Processing cost - dst1
    proc_cost = 0
    commom_len = Graph.find_common_path_len_node_main([dst1_info[0], dst2_info[0]], tmp_path_set, tmp_data_rate)
    if place_node_id_dst1 >= commom_len:
        # Check compute resoure of node enough or not
        if G_min.nodes[place_node]['com_capacity'] < all_data_rate[dst1_info[0]][dst1_info[1]][2]:
            proc_cost += float('inf')
        proc_cost += all_data_rate[dst1_info[0]][dst1_info[1]][2]

    # Processing cost - dst2
    commom_len = Graph.find_common_path_len_node_main([dst2_info[0]], tmp_path_set, tmp_data_rate)
    if place_node_id_dst2 >= commom_len:
        # Check compute resoure of node enough or not
        if G_min.nodes[place_node]['com_capacity'] < all_data_rate[dst2_info[0]][dst2_info[1]][2]:
            proc_cost += float('inf')
        proc_cost += all_data_rate[dst2_info[0]][dst2_info[1]][2]

    # print('=== ', dst1_info[0], ' ===')
    # print(tmp_path_set[dst1_info[0]])
    # print(tmp_data_rate[dst1_info[0]])
    # print('=== ', dst2_info[0], ' ===')
    # print(tmp_path_set[dst2_info[0]])
    # print(tmp_data_rate[dst2_info[0]])

    # print("merge cost = ",trans_cost, proc_cost, place_cost, "\n")
    return weight[0]*trans_cost + weight[1]*proc_cost + weight[2]*place_cost

# Update path_set, data_rate and sfc after merging for one group.
# dst_info = [dst, last_node_id_dst, place_node, next_node_id_dst]
def merge_group_info(G_min, dst1_info, dst2_info, vnf_id, group_info):
    path_set = group_info[0]
    sfc = group_info[1]
    all_data_rate = group_info[2]

    place_node = dst1_info[2]
    
    shortest_path_before_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst1_info[0]][dst1_info[1]], place_node, weight='weight')
    shortest_path_after_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst1_info[0]][dst1_info[3]], weight='weight')[1:]
    shortest_path_before_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst2_info[0]][dst2_info[1]], place_node, weight='weight')
    shortest_path_after_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst2_info[0]][dst2_info[3]], weight='weight')[1:]

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

    # path_set
    new_path_set = copy.deepcopy(path_set)

    if dst1_info[3] == -1:
        # shortest_path_after_dst1.append(dst1_info[0])
        new_path_set[dst1_info[0]][dst1_info[1]:] = shortest_path_before_dst1 + shortest_path_after_dst1
    else:
        new_path_set[dst1_info[0]][dst1_info[1]:dst1_info[3]+1] = shortest_path_before_dst1 + shortest_path_after_dst1

    if dst2_info[3] == -1:
        # shortest_path_after_dst2.append(dst2_info[0])
        new_path_set[dst2_info[0]][dst2_info[1]:] = shortest_path_before_dst2 + shortest_path_after_dst2
    else:
        new_path_set[dst2_info[0]][dst2_info[1]:dst2_info[3]+1] = shortest_path_before_dst2 + shortest_path_after_dst2

    # data_rate
    new_data_rate = update_data_rate(dst1_info, shortest_path_before_dst1, shortest_path_after_dst1, vnf_id, all_data_rate, sfc)
    new_data_rate = update_data_rate(dst2_info, shortest_path_before_dst2, shortest_path_after_dst2, vnf_id, new_data_rate, sfc)

    # sfc
    new_sfc = copy.deepcopy(sfc)
    place_node_id_dst1 = dst1_info[1] + len(shortest_path_before_dst1) - 1
    place_node_id_dst2 = dst2_info[1] + len(shortest_path_before_dst2) - 1

    new_sfc[dst1_info[0]][vnf_id] = [sfc[dst1_info[0]][vnf_id][0], place_node, place_node_id_dst1]
    if vnf_id+1 < len(sfc[dst1_info[0]]):
        new_sfc[dst1_info[0]][vnf_id+1][2] = place_node_id_dst1 + len(shortest_path_after_dst1) + 1
        if len(shortest_path_after_dst1) <= 1:
            new_sfc[dst1_info[0]][vnf_id+1][2] = new_sfc[dst1_info[0]][vnf_id][2] + (sfc[dst1_info[0]][vnf_id+1][2] - sfc[dst1_info[0]][vnf_id][2]) + len(shortest_path_after_dst1)
   
    for f in range(vnf_id+2,len(sfc[dst1_info[0]])):
        new_sfc[dst1_info[0]][f][2] = new_sfc[dst1_info[0]][f-1][2] + (sfc[dst1_info[0]][f][2] - sfc[dst1_info[0]][f-1][2])
    
    new_sfc[dst2_info[0]][vnf_id] = [sfc[dst2_info[0]][vnf_id][0], place_node, place_node_id_dst2]
    if vnf_id+1 < len(sfc[dst2_info[0]]):
        new_sfc[dst2_info[0]][vnf_id+1][2] = place_node_id_dst2 + len(shortest_path_after_dst2) + 1
        if len(shortest_path_after_dst2) <= 1:
            new_sfc[dst2_info[0]][vnf_id+1][2] = new_sfc[dst2_info[0]][vnf_id][2] + (sfc[dst2_info[0]][vnf_id+1][2] - sfc[dst2_info[0]][vnf_id][2]) + len(shortest_path_after_dst2)

    for f in range(vnf_id+2,len(sfc[dst2_info[0]])):
        new_sfc[dst2_info[0]][f][2] = new_sfc[dst2_info[0]][f-1][2] + (sfc[dst2_info[0]][f][2] - sfc[dst2_info[0]][f-1][2])
    
    
    # print('------------')
    # print(new_path_set[dst1_info[0]], new_path_set[dst2_info[0]])
    # print(new_sfc[dst1_info[0]], new_sfc[dst2_info[0]])
    # print(new_data_rate[dst1_info[0]], "\n", new_data_rate[dst2_info[0]])
    # print('-------------------------------------!!')

    return [new_path_set, new_sfc, new_data_rate, group_info[3]]

# Update data_rate after merging.
# dst_info = [dst, last_node_id_dst, place_node, next_node_id_dst] 
def update_data_rate(dst_info, shortest_path_before, shortest_path_after, vnf_id, data_rate, sfc):
    dst = dst_info[0]
    last_node_id = copy.deepcopy(dst_info[1])
    next_node_id = dst_info[3]

    new_data_rate = copy.deepcopy(data_rate)

    # print(sfc[dst][vnf_id][2], data_rate[dst], vnf_id)

    input_data = data_rate[dst][sfc[dst][vnf_id][2]-1][2]
    output_data = data_rate[dst][sfc[dst][vnf_id][2]][2]
    # print(dst)
    
    if next_node_id == -1:
        del new_data_rate[dst][last_node_id:]
    else:
        del new_data_rate[dst][last_node_id:next_node_id+1]

    input_q = data_rate[dst][sfc[dst][vnf_id][2]-1][1]
    output_q = data_rate[dst][sfc[dst][vnf_id][2]][1]

    # print("input: ", input_data, input_q, ", output: ", output_data, output_q)
    # print(vnf_id, sfc[dst])
    # print(shortest_path_before)
    # print(shortest_path_after)

    for n in shortest_path_before:
        if n == shortest_path_before[-1]:
            break
        new_data_rate[dst].insert(last_node_id, (n, input_q, input_data))
        last_node_id += 1

    new_data_rate[dst].insert(last_node_id, (dst_info[2], output_q, output_data))
    last_node_id += 1
    for n in shortest_path_after:
        if next_node_id == -1 and n == shortest_path_after[-1]: 
            break
        new_data_rate[dst].insert(last_node_id, (n, output_q, output_data))
        last_node_id += 1
    
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

    # print('---- add vnf to path begin ----')
    # print(path_set)
    # print(sfc)
    # print(data_rate)

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
                        
                        dst_info = [dst, 0, i, -1]
                        new_data_rate[dst].insert(0, (src, send_data, quality_list[send_data]))
                        new_data_rate = update_data_rate(dst_info, shortest_path_before, shortest_path_after, 0, new_data_rate, new_sfc)

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

    

    return G_min

# 1. （目前）一條一條路徑做合併
# 問題：計算成本時，無法精準計算（邊、點有無重用）
#      如何決定誰要跟誰合併？如何制定結束合併的條件？

# 2. 在全部組走完後，重建一個加權圖（計算哪個邊、點經過次數較多），以重新搜尋最短路徑盡可能共享
# 問題：如何設定權重，以避免路徑壅塞？
#      這樣做的話，分組有意義嗎？

# 3. 在全部組走完後，以最後一個有部署vnf的點為new_dst，想辦法合併前半部的路（將問題空間降低）
# 問題：如何合併前半部的路，將所有可能列出來，去選擇成本較低者嗎？