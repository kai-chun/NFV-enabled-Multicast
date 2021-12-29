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

# service = (src, dsts, sfc, data_rate)
# alpha: the weight of link transmission cost
# tree_num: the max number of trees for service request
# quality_list: all kinds of video quality
def multipath_JPR(G, service, alpha, vnf_type, tree_num, quality_list):
    N = G.nodes()

    video_type = [q for q in quality_list]

    src = service[0]
    dsts = service[1]
    sort_dsts = sorted(service[1].items(), key=lambda d: video_type.index(d[1]), reverse=True)

    beta = 1 - alpha # the weight of processing cost
    dst_list = list(d for d in dsts)
    pos = nx.spring_layout(G)

    best_quality = service[3]

    sfc = service[2]
    require_quality = set(dsts[i] for i in dsts)
    max_transcoder_num = len(require_quality) - 1
    sort_quality = sorted(require_quality, key=lambda q: video_type.index(q), reverse=True)

    # If dsts need different quality of video then place a transcoder
    if max_transcoder_num > 0: 
        for d in dst_list:
            for q in sort_quality:
                if dsts[d] != q:
                    sfc[d].append('t')
                else:
                    break

    print('--- environment ---')
    print('src = ', src, ", dst = ", dsts)
    print("sfc = ", sfc)

    # test
    if max_transcoder_num < 1:
        print('not this')
        return 0

    G_min = copy.deepcopy(G) # record the min cost network 
    multicast_path_min = nx.Graph() # record the min cost path
    min_data_rate = dict() # record the min cost data_rate

    # Record the satisfied vnf of min cost path and place node
    # index_sfc_min = {dst: {'index': the index of satisdied VNF,'place_node': [the nodes which satisdied VNFs placing]}}
    index_sfc_min = dict((d,{'index': -1, 'place_node':[]}) for d in dst_list) 

    for node in N:
        # copy G to G_new, and update the weight of edges
        G_new = copy.deepcopy(G)
        nx.set_edge_attributes(G_new, {e: {'weight': Graph.cal_link_weight(G_new,e,alpha,beta,quality_list[best_quality])} for e in G_new.edges})

        # generate metric closure
        G_metric_closure = nx.algorithms.approximation.metric_closure(G_new,weight='weight')

        # generate the set of {src, dsts, n} in G_new
        G_keyNode = nx.Graph()
        G_keyNode.add_nodes_from([src,node])
        G_keyNode.add_nodes_from(dst_list)

        for n1,n2,d in G_metric_closure.edges(data=True):
            if n1 in G_keyNode and n2 in G_keyNode:
                G_keyNode.add_edge(n1,n2,weight=d['distance'])

        # generate MST
        T = nx.minimum_spanning_tree(G_keyNode,weight='distance')

        # replace edges in MST with corresponding paths in G_new
        keynode_path = nx.Graph(T)
        for e in T.edges:
            if e not in G:
                keynode_path.remove_edge(e[0],e[1])
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_new, e[0],e[1], weight='weight')
                for j in shortest_path:
                    for i in range(len(shortest_path)-1):
                        keynode_path.add_edge(shortest_path[i],shortest_path[i+1],data_rate=0)

        #Graph.printGraph(keynode_path, pos, 'keynode_path_'+str(node), service, 'data_rate')

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
            shortest_path_set[dst] = shortest_path
            
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

            # print(shortest_path_set)
            # print('------------------------')
            # Iterative the node in shortes_path from src to dst
            for i in range(len(shortest_path)-1):
                # Update the bandwidth of link
                e = (shortest_path[i],shortest_path[i+1])
                Graph.add_new_edge(tmp_G, tmp_multicast_path, shortest_path_set, dst, e, data_rate[dst][-1], video_type)

                # Don't place vnf on src and dst
                if shortest_path[i] != src and shortest_path[i] != dst:
                    j = shortest_path[i]

                    # Find node to place vnf until finishing sfc
                    while index_sfc[dst]['index'] < len(sfc[dst])-1:
                        vnf = sfc[dst][index_sfc[dst]['index']+1]

                        # There are vnf instance on node j
                        if vnf in tmp_G.nodes[j]['vnf']:
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)

                            output_q = sort_quality[index_sfc[dst]['index']+1]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((j,output_q,update_data_rate))
                            continue
                        
                        # Check if the node has enough resource to place instance
                        if tmp_G.nodes[j]['capacity'] > vnf_type[vnf]:
                            tmp_G.nodes[j]['vnf'].append(vnf)
                            tmp_multicast_path.nodes[j]['vnf'].append(vnf)
                            tmp_G.nodes[j]['capacity'] -= vnf_type[vnf]
                            index_sfc[dst]['index'] += 1
                            index_sfc[dst]['place_node'].append(j)
                            
                            output_q = sort_quality[index_sfc[dst]['index']+1]
                            update_data_rate = exp_set.cal_transcode_bitrate(data_rate[dst][-1][2], data_rate[dst][-1][1], output_q)
                            data_rate[dst].append((j,output_q,update_data_rate))
                        else:
                            break

            # print('--- tmp path ---')
            # print(tmp_multicast_path.nodes(data=True))
            # print(tmp_multicast_path.edges(data=True))
            Graph.printGraph(tmp_multicast_path, pos, 'tmp', service, 'data_rate')

            # print(Graph.cal_cost(tmp_G, [tmp_multicast_path], alpha, vnf_type), Graph.count_instance(tmp_multicast_path))
        #print('vnf ok')
        # Greedy selection
        if nx.classes.function.is_empty(multicast_path_min):
            multicast_path_min = copy.deepcopy(tmp_multicast_path)
            G_min = copy.deepcopy(tmp_G)
            index_sfc_min = copy.deepcopy(index_sfc)
            min_data_rate = copy.deepcopy(data_rate)
        elif Graph.count_instance(tmp_multicast_path) == Graph.count_instance(multicast_path_min) and \
            Graph.cal_cost(tmp_G, [tmp_multicast_path], alpha, vnf_type) < Graph.cal_cost(G_min, [multicast_path_min], alpha, vnf_type):
                multicast_path_min = copy.deepcopy(tmp_multicast_path)
                G_min = copy.deepcopy(tmp_G)
                index_sfc_min = copy.deepcopy(index_sfc)
                min_data_rate = copy.deepcopy(data_rate)
        elif Graph.count_instance(tmp_multicast_path) > Graph.count_instance(multicast_path_min):
                multicast_path_min = copy.deepcopy(tmp_multicast_path)
                G_min = copy.deepcopy(tmp_G)
                index_sfc_min = copy.deepcopy(index_sfc)
                min_data_rate = copy.deepcopy(data_rate)

    print('--- final ---')
    print(multicast_path_min.nodes(data=True))
    #print(Graph.cal_cost(G_min, [multicast_path_min], alpha, vnf_type), Graph.count_instance(tmp_multicast_path))
    print(multicast_path_min.edges(data=True))

    print("index_sfc = ", index_sfc_min)
    print("shortest path = ", shortest_path_set)

    Graph.printGraph(G_min, pos, 'G_min_miss_vnf', service, 'bandwidth')
    Graph.printGraph(multicast_path_min, pos, 'path_miss_vnf', service, 'data_rate')
    
    # If dsts have same quality of video then return the path without VNF
    if max_transcoder_num == 0:
        print('no vnf')
        return (G_min, multicast_path_min)

    '''
    A corrective subroutine that places the missing NF instances 
    on the closest NFV node from the multicast topology.
    '''
    missing_vnf_dsts = []
    for dst in index_sfc_min: # find all dst which missing vnf 
        if index_sfc_min[dst]['index'] < len(sfc[dst])-1:
            missing_vnf_dsts.append(dst)
    
    # Remove unnecessary edges: the path from last vnf node to dst
    update_shortest_path_set = copy.deepcopy(shortest_path_set)
    for dst in missing_vnf_dsts:
        last_place_node = index_sfc_min[dst]['place_node'][-1] if index_sfc_min[dst]['index'] > -1 else -1
        last_node_index = shortest_path_set[dst].index(last_place_node) if last_place_node != -1 else 0
        del update_shortest_path_set[dst][last_node_index+1:]

    # Rebuilding the multicast path with update_shortest_path_set
    G_min.remove_edges_from(G_min.edges())
    for n1,n2,d in G.edges(data=True):
        G_min.add_edge(n1,n2,data_rate=0,bandwidth=d['bandwidth'])
    
    multicast_path_min.remove_edges_from(multicast_path_min.edges())
    for d in sort_dsts:
        dst = d[0]
        for i in range(len(update_shortest_path_set[dst])-1):
            e = (update_shortest_path_set[dst][i], update_shortest_path_set[dst][i+1])
            print(i)
            if update_shortest_path_set[dst][i] != data_rate[dst][i][0]:
                print("no!!!!!!")
            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate[dst][i], video_type)

    #print(multicast_path_min.nodes(data=True))
    #print(multicast_path_min.edges(data=True))

    Graph.printGraph(G, pos, 'G', service, 'bandwidth')
    Graph.printGraph(G_min, pos, 'G_min_remove', service, 'bandwidth')
    Graph.printGraph(multicast_path_min, pos, 'path_remove', service, 'data_rate')

    print("update shortest path = ", update_shortest_path_set)
    print(data_rate)
    return 0
    # Place vnf nearest the common path
    for dst in missing_vnf_dsts:
        find_distance = 0 # the length range of finding node away from last_node that placing VNF
        place_flag = 0 # whether placing VNF or not
        
        while index_sfc_min[dst]['index'] < len(sfc[dst])-1:
            last_node = update_shortest_path_set[dst][-1]
            if place_flag == 0:
                find_distance += 1
            else:
                find_distance = 1
                place_flag = 0
            alternate_nodes_len = nx.single_source_shortest_path_length(G, last_node, find_distance)
            alternate_nodes = list(n for n in nx.single_source_shortest_path_length(G, last_node, find_distance))
            
            for i in alternate_nodes:
                if place_flag == 1:
                    break
                # If find_distance > 1, the node that distance = 1 have been searched
                # it don't have to consider the node again
                if find_distance > 1 and alternate_nodes_len[i] == find_distance-1:
                    continue

                last_node = update_shortest_path_set[dst][-1]
                shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, i, weight='data_rate')
                
                vnf = sfc[dst][index_sfc_min[dst]['index']+1]
                if i == last_node or i == src or i == dst:
                    continue

                # There are vnf instance on node i
                while vnf in G_min.nodes[i]['vnf']:
                    index_sfc_min[dst]['index'] += 1
                    index_sfc_min[dst]['place_node'].append(i)
                    place_flag = 1

                    if find_distance == 1:
                        update_shortest_path_set[dst].append(i)
                        if i not in multicast_path_min:
                            multicast_path_min.add_node(i, vnf=[])
                        e = (last_node, i)
                        # If previous_node to i don't have path then build it
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
                    else:
                        for j in range(len(shortest_path)-1):
                            update_shortest_path_set[dst].append(shortest_path[j+1])
                            if j not in multicast_path_min:
                                multicast_path_min.add_node(shortest_path[j], vnf=[])
                            e = (shortest_path[j], shortest_path[j+1])
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)

                    if index_sfc_min[dst]['index'] == len(sfc[dst])-1: # finish sfc
                        break
                    vnf = sfc[dst][index_sfc_min[dst]['index']+1]
                    last_node = i

                if index_sfc_min[dst]['index'] == len(sfc[dst])-1: # finish sfc
                    break

                # Check if node has enough resource to place instance
                while G_min.nodes[i]['capacity'] > vnf_type[vnf]:
                    G_min.nodes[i]['vnf'].append(vnf)
                    if i not in multicast_path_min:
                        multicast_path_min.add_node(i, vnf=[])
                    multicast_path_min.nodes[i]['vnf'].append(vnf)
                    G_min.nodes[i]['capacity'] -= vnf_type[vnf]
                    index_sfc_min[dst]['index'] += 1
                    index_sfc_min[dst]['place_node'].append(i)
                    place_flag = 1

                    if find_distance == 1:
                        update_shortest_path_set[dst].append(i)
                        e = (last_node, i)
                        # If previous_node to i don't have path then build it
                        Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
                    else:
                        for j in range(len(shortest_path)-1):
                            update_shortest_path_set[dst].append(shortest_path[j+1])
                            e = (shortest_path[j], shortest_path[j+1])
                            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
                    
                    if index_sfc_min[dst]['index'] == len(sfc[dst])-1: # finish sfc
                        break
                    vnf = sfc[dst][index_sfc_min[dst]['index']+1]
                    last_node = i
                
                if index_sfc_min[dst]['index'] == len(sfc[dst])-1: # finish sfc
                    break

    print("final update shortest path = ", update_shortest_path_set)
    Graph.printGraph(multicast_path_min, pos, 'tmp', service, 'data_rate')
        
    # Construct the path from the last_placement_node to dst and ignore bandwidth constraint
    for dst in missing_vnf_dsts:
        last_node = update_shortest_path_set[dst][-1]
        shortest_path = nx.algorithms.shortest_paths.dijkstra_path(G_min, last_node, dst, weight='data_rate')
        # print("shortest path from last_node to dst = ", shortest_path)
        for i,j in enumerate(shortest_path):
            if i == 0:
                continue
            if j not in multicast_path_min:
                multicast_path_min.add_node(j, vnf=[])
            e = (last_node, j)
            Graph.add_new_edge(G_min, multicast_path_min, update_shortest_path_set, dst, e, data_rate)
            update_shortest_path_set[dst].append(j)
            last_node = j

    print("final_path", update_shortest_path_set)
    Graph.printGraph(G_min, pos, 'G_min_update', service, 'bandwidth')
    Graph.printGraph(multicast_path_min, pos, 'path_update', service, 'data_rate')

    Graph.printGraph(G_new, pos, 'G_new', service, 'weight')
    Graph.printGraph(G_metric_closure, pos, 'G_metric', service, 'distance')
    Graph.printGraph(G_keyNode, pos, 'G_key', service, 'weight')
    Graph.printGraph(T, pos, 'T', service, 'weight')

    '''
    Check whether each path in multicast_path satisfy the link transmission rate requirement.
    If the single-path solution is infeasible, it will be extended to the multipath solution.
    '''
    G_final = copy.deepcopy(G)
    for m in multicast_path_min:
        G_final.nodes[m]['vnf'] = multicast_path_min.nodes[m]['vnf']
    path_final = copy.deepcopy(multicast_path_min)
    path_final.remove_edges_from(multicast_path_min.edges)

    use_tree = [] # Record all multipath trees

    for dst in update_shortest_path_set:
        dst_path = update_shortest_path_set[dst]
        for i in range(len(dst_path)-1):
            success_flag = False
            tree_list = list(nx.all_simple_paths(G, source=dst_path[i], target=dst_path[i+1]))
            tree_list_num = len(tree_list)

            # Sort all candidate paths in a descending order 
            # based on the amount of residual transmission resource.
            tree_resource = dict()
            for k in range(tree_list_num):
                tree_resource[k] = 0
                for m in range(len(tree_list[k])-1):
                    tree_resource[k] += G.edges[tree_list[k][m],tree_list[k][m+1]]['bandwidth']
            tree_resource_sort = sorted(tree_resource.items(), key=lambda x:x[1], reverse=True)
            
            for j in range(tree_num-1):
                tmp_index = min(j, tree_list_num)
                tmp_path = tree_list[tree_resource_sort[tmp_index][0]]

                min_bandwidth = sys.maxsize
                for m in range(len(tmp_path)-1):
                    min_bandwidth = min(G_final.edges[tmp_path[m],tmp_path[m+1]]['bandwidth'], min_bandwidth)
                    
                # Check if edges of tmp_path have enough resource to transmission data
                if min_bandwidth > data_rate:
                    use_tree.append(tmp_path)
                    for m in tmp_path:
                        if m not in path_final:
                            path_final.add_node(m, vnf=[])
                    for m in range(len(tmp_path)-1):
                        e = (tmp_path[m],tmp_path[m+1])

                        allocate_data_rate = data_rate / (tmp_index+1) # averagely allocate data rate to each tree
                        Graph.add_new_edge(G_final, path_final, update_shortest_path_set, dst, e, allocate_data_rate)
                    success_flag = True
                    break
            
            if success_flag == True:
                break
            
    #print(use_tree)
    #Graph.printGraph(G_final, pos, 'G_final', service, 'bandwidth')
    #Graph.printGraph(path_final, pos, 'path_final', service, 'data_rate')
    return (G_final, path_final)