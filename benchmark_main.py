import random
import networkx as nx
from networkx.linalg.algebraicconnectivity import _get_fiedler_func

import Graph
import experience_setting as exp_set
import grouping
import multipath

### Parameters
alpha = 0.6
beta = 1 - alpha
dst_num = random.randint(8, 10)
vnf_num = random.randint(1,1)
vnf_type = {'t':1}#,'b':1,'c':1,'d':1,'e':1}
quality_list = {'224p': 0.25, '360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
video_type = [q for q in quality_list]
user_limit = 1
tree_num = 3

# Generate topology
G = exp_set.create_topology(20, 0.6)
print('graph ok')
N = G.nodes()
pos = nx.spring_layout(G)

# Generate service
src = random.choice(list(N))
dsts = exp_set.create_client(dst_num, N, src, video_type)
dst_list = list(d for d in dsts)

# Grouping
group_list = grouping.k_means(pos, dsts, video_type, user_limit)
print('grouping ok')
for i in group_list:
    group = group_list[i]
    group_info = dict()
    sfc = dict()
    for g in group:
        group_info[g] = dsts[g]
        sfc[g] = []
    
    best_quality = video_type[max(video_type.index(group_info[g]) for g in group_info)]
    service = (src, group_info, sfc, best_quality)
    G_ans = multipath.multipath_JPR(G, service, alpha, vnf_type, tree_num, quality_list)
    break
    #Graph.printGraph(G_ans[0], pos, 'G_final_'+str(i), service, 'bandwidth')
    #Graph.printGraph(G_ans[1], pos, 'path_final_'+str(i), service, 'data_rate')
