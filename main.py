import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import clique

import Graph
import experience_setting as exp_set
import grouping

### Parameters
alpha = 0.6
beta = 1 - alpha
dst_num = random.randint(5, 10)
vnf_num = random.randint(2, 3)
vnf_type = {'t': 1} #,'b':1,'c':1,'d':1,'e':1}
quality_list = {'224p': 0.25, '360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
user_limit = 1

# Generate topology
G = exp_set.create_topology(20, 0.2)
N = G.nodes()
pos = nx.spring_layout(G)

# Generate service
src = random.choice(list(N))
dsts = exp_set.create_client(dst_num, N, src, quality_list)
sfc = ['t']#sfc = random.sample(sorted(vnf_type), vnf_num)
dst_list = list(d for d in dsts)
service = (src, dsts, sfc, quality_list['1080p'])

# Grouping
group_list = grouping.k_means(pos, dsts, quality_list, user_limit)

for i in group_list:
    group = group_list[i]
    group_info = []
    tree_list = []
    multicast_tree = nx.Graph()
    
    for g in group:
        group_info.append((g,dsts[g]))

        path_nodes = nx.algorithms.shortest_paths.dijkstra_path(G, src, g, weight='weight')
        path_edges = []
        for j in range(len(path_nodes)-1):
            e = (path_nodes[j], path_nodes[j+1])
            path_edges.append(e)
        
        multicast_tree.add_edges_from(path_edges)

    print(multicast_tree) 
    '''要把 G 的頻寬加到 multicast_tree 的邊上，才能把圖印出來；但後面方法會用到頻寬嗎？'''
    #Graph.printGraph(multicast_tree, pos, 'test', service, 'bandwidth')

Graph.printGraph(G, pos, 'G', service, 'bandwidth')

'''
分組之後，針對同一組的人決定發送品質、最大需要多少個轉碼器
假設有三種品質，那最多需要兩個轉碼器。
接著對所有 src 到 dst 找最短路徑，找到最簡單的 tree
對 tree 做 greedy 找放置轉碼器的位置
這樣就找到每組的部署轉碼器位置及路由路徑

接著找最近的 group
對這兩個組找最好的品質，並對這兩個組的上游找可以放轉碼器的部分

這樣分組是不是純粹以品質分就好不要考慮地理位置？
'''