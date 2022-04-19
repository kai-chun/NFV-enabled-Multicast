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
from inspect import trace
from struct import pack
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
        G_tmp = copy.deepcopy(G_min)
        remove_edges = list((n1,n2) for n1,n2,dic in G_min.edges(data=True) if dic['bandwidth'] < source_data_size)
        G_tmp.remove_edges_from(remove_edges)

        try:
            shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_tmp, src, dst, weight='data_rate')
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
    reverse_sort = group_info[4]
    reverse_sort.reverse()

    video_type = [q for q in quality_list]
    video_type.reverse()
    src = path_set[list(path_set)[0]][0]

    ### 用一個矩陣記錄兩兩路徑間合併某節點的成本，選最低者
    ### 過程中用 threshold（hop數）降低複雜度
    new_group_info = copy.deepcopy(group_info)[1:]
    path_set = new_group_info[0]
    sfc = new_group_info[1]
    data_rate = new_group_info[2]
    path_length_threshold = 5 #len(G_min.edges)
    
    max_sfc_len = len(sfc[reverse_sort[0][0]])
    # print('------------')
    # print(path_set)
    # print(sfc)
    # print(data_rate)
    # print('-----------')
    for vnf_id in range(max_sfc_len):
        print("=========== vnf_id = ", vnf_id, " ===========")
        isFinish = False
        count = 0
        while isFinish == False:
            path_set = new_group_info[0]
            sfc = new_group_info[1]
            data_rate = new_group_info[2]
            print("------ count = ", count, " ------")
            print(path_set)
            print(sfc)
            print(data_rate)
            print('-----------')
            count += 1
            cost_metric = dict()
            for i in range(len(reverse_sort)):
                dst1 = reverse_sort[i][0]
                for j in range(i+1, len(reverse_sort)):
                    dst2 = reverse_sort[j][0]

                    if vnf_id < len(sfc[dst1]) and vnf_id < len(sfc[dst2]):
                        shortest_path_between_vnfs = nx.algorithms.shortest_paths.dijkstra_path(G_min, sfc[dst1][vnf_id][1], sfc[dst2][vnf_id][1], weight='data_rate')
                        if len(shortest_path_between_vnfs) - 2 > path_length_threshold:
                            cost_metric[(dst1,dst2)] = (0, -1, [], [])
                        else:
                            last_node_id_dst1 = sfc[dst1][vnf_id-1][2]
                            last_node_id_dst2 = sfc[dst2][vnf_id-1][2]

                            if vnf_id == 0: # then last_node is src
                                last_node_id_dst1 = 0 
                                last_node_id_dst2 = 0

                            orig_cost = cal_cost_origin(G_min, dst1, last_node_id_dst1, vnf_id, new_group_info, dst1) + cal_cost_origin(G_min, dst2, last_node_id_dst2, vnf_id, new_group_info, dst1)
                            merge_cost = orig_cost
                            merge_node = -1

                            if vnf_id <= len(sfc[dst1])-1:
                                next_node_id_dst1 = sfc[dst1][vnf_id][2]
                            else:
                                next_node_id_dst1 = -1
                            
                            if vnf_id <= len(sfc[dst2])-1:
                                next_node_id_dst2 = sfc[dst2][vnf_id][2]
                            else:
                                next_node_id_dst2 = -1

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

                                total_cost = cal_cost_merge(G_min, dst1_info, dst2_info, vnf_id, new_group_info, video_type)

                                if total_cost < merge_cost:
                                    merge_cost = total_cost
                                    merge_node = m
                            # print('------------ ', dst2, ' -------------')
                            dst1_info = [dst1, last_node_id_dst1, merge_node, next_node_id_dst1]
                            dst2_info = [dst2, last_node_id_dst2, merge_node, next_node_id_dst2]
                            cost_metric[(dst1,dst2)] = (round(orig_cost - merge_cost, 2), merge_node, dst1_info, dst2_info)
                
            print("cost_metric: ", cost_metric)
            # Choose the maximum save_cost (orig_cost - merge_cost) and merge the paths with merge_node.
            merge_result = max(cost_metric, key=lambda k: cost_metric[k][0])
            if cost_metric[merge_result][0] <= 0:
                isFinish = True
                print("no merge")
                break
            print(merge_result, cost_metric[merge_result])
            
            print('------------')
            print(path_set[merge_result[0]], path_set[merge_result[1]])
            print(sfc[merge_result[0]], sfc[merge_result[1]])
            print(data_rate[merge_result[0]], data_rate[merge_result[1]])
            print('-----------')

            dst1_info = cost_metric[merge_result][2]
            dst2_info = cost_metric[merge_result][3]

            new_group_info = update_group_info(G_min, dst1_info, dst2_info, vnf_id, new_group_info, video_type)
        
    return new_group_info

# Calculate total cost from src_vnf to dst_vnf.
def cal_cost_origin(G_min, dst, last_node_id, index, group_info, another_dst):
    path_set = group_info[0]
    sfc = group_info[1]
    all_data_rate = group_info[2]

    vnf_info = group_info[1][dst]
    data_rate = all_data_rate[dst]

    if index <= len(vnf_info)-1:
        dst_vnf_id = vnf_info[index][2]
    else:
        dst_vnf_id = len(data_rate)

    place_node = path_set[dst][dst_vnf_id]
    
    # Transimission cost
    trans_cost = 0
    commom_len = Graph.find_common_path_len_edge(dst, path_set, all_data_rate) 
    if dst_vnf_id >= commom_len: # If not multicast link
        max_len = max(commom_len-1, last_node_id)
        trans_cost += data_rate[last_node_id][2] * (dst_vnf_id - max_len)
    
    # Placing cost
    place_cost = 1
    if place_node == dst:
        place_cost = 0
    else:
        for d in sfc:
            if place_cost == 0: break
            for v in sfc[d]:
                if d != dst and d != another_dst and v[1] == place_node:
                    place_cost = 0
                    break
                if (d == dst or d == another_dst) and v[1] == place_node and v[2] != dst_vnf_id:
                    place_cost = 0
                    break

    # Processing cost
    proc_cost = 0
    commom_len = Graph.find_common_path_len_node_main([dst, another_dst], path_set, all_data_rate)
    if dst_vnf_id >= commom_len and place_node != dst:
        proc_cost = data_rate[last_node_id][2]
    
    # print(dst, "orig cost = ",trans_cost, place_cost, proc_cost, "\n")
    return trans_cost + place_cost + proc_cost
        
# Calculate total cost from src_vnf to dst_vnf.
# dst_info = [dst, last_node_id_dst, place_node, next_node_id_dst]
def cal_cost_merge(G_min, dst1_info, dst2_info, vnf_id, group_info, video_type):
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
    
    shortest_path_before_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst1_info[0]][dst1_info[1]], place_node, weight='data_rate')
    shortest_path_after_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst1_info[0]][dst1_info[3]], weight='data_rate')[1:]
    shortest_path_before_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst2_info[0]][dst2_info[1]], place_node, weight='data_rate')
    shortest_path_after_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst2_info[0]][dst2_info[3]], weight='data_rate')[1:]

    tmp_path_set[dst1_info[0]][dst1_info[1]:] = shortest_path_before_dst1
    tmp_path_set[dst2_info[0]][dst2_info[1]:] = []
    tmp_data_rate = update_data_rate(dst1_info, shortest_path_before_dst1, shortest_path_after_dst1, vnf_id, all_data_rate, sfc, video_type)
    tmp_data_rate = update_data_rate(dst2_info, shortest_path_before_dst2, shortest_path_after_dst2, vnf_id, tmp_data_rate, sfc, video_type)

    place_node_id_dst1 = len(tmp_path_set[dst1_info[0]]) - 1
    commom_len_dst1 = Graph.find_common_path_len_edge(dst1_info[0], tmp_path_set, tmp_data_rate) 
    if place_node_id_dst1 >= commom_len_dst1: # If not multicast link
        max_len = max(commom_len_dst1-1, dst1_info[1])
        trans_cost += all_data_rate[dst1_info[0]][dst1_info[1]][2] * (place_node_id_dst1 - max_len)
    
    # Transimission cost - dst2, before place_node
    tmp_path_set[dst2_info[0]][dst2_info[1]:] = shortest_path_before_dst2
    
    place_node_id_dst2 = len(tmp_path_set[dst2_info[0]]) - 1
    commom_len_dst2 = Graph.find_common_path_len_edge(dst2_info[0], tmp_path_set, tmp_data_rate) 
    if place_node_id_dst2 >= commom_len_dst2: # If not multicast link
        max_len = max(commom_len_dst2-1, dst2_info[1])
        trans_cost += all_data_rate[dst2_info[0]][dst2_info[1]][2] * (place_node_id_dst2 - max_len)

    # Transimission cost - dst1, after place_node
    tmp_path_set[dst1_info[0]].extend(shortest_path_after_dst1)
    commom_len_dst1 = Graph.find_common_path_len_edge(dst1_info[0], tmp_path_set, tmp_data_rate) 
    if len(tmp_path_set[dst1_info[0]]) - 1 >= commom_len_dst1: # If not multicast link
        max_len = max(commom_len_dst1-1, place_node_id_dst1)
        trans_cost += all_data_rate[dst1_info[0]][sfc[dst1_info[0]][vnf_id][2]][2] * (len(tmp_path_set[dst1_info[0]]) - 1 - max_len)

    # Transimission cost - dst2, after place_node
    tmp_path_set[dst2_info[0]].extend(shortest_path_after_dst2)
    commom_len_dst2 = Graph.find_common_path_len_edge(dst2_info[0], tmp_path_set, tmp_data_rate) 
    if len(tmp_path_set[dst2_info[0]]) - 1 >= commom_len_dst2: # If not multicast link
        max_len = max(commom_len_dst2-1, place_node_id_dst2)
        trans_cost += all_data_rate[dst2_info[0]][sfc[dst2_info[0]][vnf_id][2]][2] * (len(tmp_path_set[dst2_info[0]]) - 1 - max_len)

    # Placing cost
    place_cost = 1
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
        proc_cost += all_data_rate[dst1_info[0]][dst1_info[1]][2]

    # Processing cost - dst2
    commom_len = Graph.find_common_path_len_node_main([dst2_info[0]], tmp_path_set, tmp_data_rate)
    if place_node_id_dst2 >= commom_len:
        proc_cost += all_data_rate[dst2_info[0]][dst2_info[1]][2]

    # print('=== ', dst1_info[0], ' ===')
    # print(tmp_path_set[dst1_info[0]])
    # print(tmp_data_rate[dst1_info[0]])
    # print('=== ', dst2_info[0], ' ===')
    # print(tmp_path_set[dst2_info[0]])
    # print(tmp_data_rate[dst2_info[0]])

    # print("merge cost = ",trans_cost, place_cost, proc_cost, "\n")
    return trans_cost + place_cost + proc_cost

# Update path_set, data_rate and sfc after merging.
# dst_info = [dst, last_node_id_dst, place_node, next_node_id_dst]
def update_group_info(G_min, dst1_info, dst2_info, vnf_id, group_info, video_type):
    path_set = group_info[0]
    sfc = group_info[1]
    all_data_rate = group_info[2]

    place_node = dst1_info[2]
    
    shortest_path_before_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst1_info[0]][dst1_info[1]], place_node, weight='data_rate')
    shortest_path_after_dst1 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst1_info[0]][dst1_info[3]], weight='data_rate')[1:]
    shortest_path_before_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, path_set[dst2_info[0]][dst2_info[1]], place_node, weight='data_rate')
    shortest_path_after_dst2 = nx.algorithms.shortest_paths.dijkstra_path(G_min, place_node, path_set[dst2_info[0]][dst2_info[3]], weight='data_rate')[1:]

    print("--- update_group_info ---")
    print('======= dst1 ========')
    print(shortest_path_before_dst1)
    print(shortest_path_after_dst1)
    print('======= dst2 ========')
    print(shortest_path_before_dst2)
    print(shortest_path_after_dst2)

    # path_set
    new_path_set = copy.deepcopy(path_set)

    if dst1_info[3] == -1:
        shortest_path_after_dst1.append(dst1_info[0])
        new_path_set[dst1_info[0]][dst1_info[1]:] = shortest_path_before_dst1 + shortest_path_after_dst1
    else:
        new_path_set[dst1_info[0]][dst1_info[1]:dst1_info[3]+1] = shortest_path_before_dst1 + shortest_path_after_dst1

    if dst2_info[3] == -1:
        shortest_path_after_dst2.append(dst2_info[0])
        new_path_set[dst2_info[0]][dst2_info[1]:] = shortest_path_before_dst2 + shortest_path_after_dst2
    else:
        new_path_set[dst2_info[0]][dst2_info[1]:dst2_info[3]+1] = shortest_path_before_dst2 + shortest_path_after_dst2

    # data_rate
    new_data_rate = update_data_rate(dst1_info, shortest_path_before_dst1, shortest_path_after_dst1, vnf_id, all_data_rate, sfc, video_type)
    new_data_rate = update_data_rate(dst2_info, shortest_path_before_dst2, shortest_path_after_dst2, vnf_id, new_data_rate, sfc, video_type)

    # sfc
    new_sfc = copy.deepcopy(sfc)
    place_node_id_dst1 = dst1_info[1] + len(shortest_path_before_dst1) - 1
    place_node_id_dst2 = dst2_info[1] + len(shortest_path_before_dst2) - 1

    new_sfc[dst1_info[0]][vnf_id] = [sfc[dst1_info[0]][vnf_id][0], place_node, place_node_id_dst1]
    if vnf_id+1 < len(sfc[dst1_info[0]]):
        new_sfc[dst1_info[0]][vnf_id+1][2] = place_node_id_dst1 + len(shortest_path_after_dst1) + 1
    for f in range(vnf_id+2,len(sfc[dst1_info[0]])):
        new_sfc[dst1_info[0]][f][2] = new_sfc[dst1_info[0]][f-1][2] + (sfc[dst1_info[0]][f][2] - sfc[dst1_info[0]][f-1][2])
    
    new_sfc[dst2_info[0]][vnf_id] = [sfc[dst2_info[0]][vnf_id][0], place_node, place_node_id_dst2]
    if vnf_id+1 < len(sfc[dst2_info[0]]):
        new_sfc[dst2_info[0]][vnf_id+1][2] = place_node_id_dst2 + len(shortest_path_after_dst2) + 1
    for f in range(vnf_id+2,len(sfc[dst2_info[0]])):
        new_sfc[dst2_info[0]][f][2] = new_sfc[dst2_info[0]][f-1][2] + (sfc[dst2_info[0]][f][2] - sfc[dst2_info[0]][f-1][2])

    print('------------')
    print(new_path_set[dst1_info[0]], new_path_set[dst2_info[0]])
    print(new_sfc[dst1_info[0]], new_sfc[dst2_info[0]])
    print(new_data_rate[dst1_info[0]], new_data_rate[dst2_info[0]])
    print('------------')

    return [new_path_set, new_sfc, new_data_rate]

# Update data_rate after merging.
# dst_info = [dst, last_node_id_dst, place_node, next_node_id_dst] 
def update_data_rate(dst_info, shortest_path_before, shortest_path_after, vnf_id, data_rate, sfc, video_type):
    dst = dst_info[0]
    last_node_id = copy.deepcopy(dst_info[1])
    next_node_id = dst_info[3]

    new_data_rate = copy.deepcopy(data_rate)

    input_data = data_rate[dst][last_node_id][2]
    output_data = data_rate[dst][sfc[dst][vnf_id][2]][2]
    
    if next_node_id == -1:
        del new_data_rate[dst][last_node_id:]
    else:
        del new_data_rate[dst][last_node_id:next_node_id+1]

    for n in shortest_path_before:
        if n == shortest_path_before[-1]:
            break
        new_data_rate[dst].insert(last_node_id, (n, video_type[int(sfc[dst][vnf_id][0][-2])], input_data))
        last_node_id += 1

    new_data_rate[dst].insert(last_node_id, (dst_info[2], video_type[int(sfc[dst][vnf_id][0][-1])], output_data))
    last_node_id += 1
    for n in shortest_path_after:
        new_data_rate[dst].insert(last_node_id, (n, video_type[int(sfc[dst][vnf_id][0][-1])], output_data))
        last_node_id += 1

    return new_data_rate

# Use new_path_info to create new path on G.
def update_graph(G, new_path_info, sort_dsts):
    path_set = new_path_info[0]
    sfc = new_path_info[1]
    data_rate = new_path_info[2]
    
    # print('=============')
    # print(sort_dsts)
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
            # print(e)
            Graph.add_new_edge(G_min, multicast_path, tmp_list, dst, e, data_rate[dst][i-1], data_rate)

    #print(multicast_path.nodes(data=True))
    return G_min

# 1. （目前）一條一條路徑做合併
# 問題：計算成本時，無法精準計算（邊、點有無重用）
#      如何決定誰要跟誰合併？如何制定結束合併的條件？

# 2. 在全部組走完後，重建一個加權圖（計算哪個邊、點經過次數較多），以重新搜尋最短路徑盡可能共享
# 問題：如何設定權重，以避免路徑壅塞？
#      這樣做的話，分組有意義嗎？

# 3. 在全部組走完後，以最後一個有部署vnf的點為new_dst，想辦法合併前半部的路（將問題空間降低）
# 問題：如何合併前半部的路，將所有可能列出來，去選擇成本較低者嗎？