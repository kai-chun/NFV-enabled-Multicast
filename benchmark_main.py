import copy
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import Graph
import experience_setting as exp_set
import grouping
import benchmark_JPR
import benchmark_firstFit

'''
1. 寫一個簡單版對照組
2. 調整各種參數（拓墣、節點數[20-100,200]、使用者比例），印出數據
'''

### Parameters
alpha = 0.6
beta = 1 - alpha

node_num = 12
edge_prob = 0.6

dst_ratio = 0.5
dst_num = int(np.ceil(node_num * dst_ratio))
vnf_num = random.randint(1,1)
vnf_type = {'t':1}#,'b':1,'c':1,'d':1,'e':1}
quality_list = {'360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
video_type = [q for q in quality_list]
user_limit = 1
tree_num = 3
exp_num = 50

def main():
    graph_exp = {'node_num': node_num, 'edge_prob': edge_prob}
    service_exp = {'dst_num': dst_num, 'video_type': video_type}

    for i in range(exp_num):
        exp_set.generate_exp(graph_exp, service_exp, i)

    exp_data_JPR = []

    for i in range(exp_num):
        # Read Graph
        input_graph = exp_set.read_exp_graph(i)
        G_JPR = input_graph[0]
        G_firstFit = input_graph[0]
        pos = input_graph[1]

        # nx.draw(G, pos)
        # plt.savefig('exp_setting/G.png')
        # plt.close()
    
        input_service = exp_set.read_exp_service(i)
        src = input_service[0]
        dsts = input_service[1]

        #print('graph ok')

        # Grouping
        group_list = grouping.k_means(pos, dsts, video_type, user_limit)
        #print('grouping ok')

        # Find multicast path for all groups
        use_tree = []
        failed_dsts = []
        for i in group_list:
            group = group_list[i]
            group_info = dict()
            sfc = dict()
            for g in group:
                group_info[g] = dsts[g]
                sfc[g] = []
            
            best_quality = video_type[max(video_type.index(group_info[g]) for g in group_info)]
            service = (src, group_info, sfc, best_quality)
            G_JPR_ans = benchmark_JPR.search_multipath(G_JPR, service, alpha, vnf_type, quality_list, isReuse=True)
            G_firstFit_ans = benchmark_firstFit.search_multipath(G_firstFit, service)
            
            #Graph.printGraph(G_ans[0], pos, 'G_final_'+str(i), service, 'bandwidth')
            #Graph.printGraph(G_ans[1], pos, 'path_final_'+str(i), service, 'data_rate')
            G_JPR = copy.deepcopy(G_JPR_ans[0])  
            if nx.classes.function.is_empty(G_JPR_ans[1]):
                failed_dsts += group
            else:
                use_tree.append(G_JPR_ans[1])
            
        total_cost_JPR = Graph.cal_total_cost(G_JPR)
        exp_data_JPR.append(total_cost_JPR)
        #print("total_cost = ", total_cost)
        #print("failed dsts = ", failed_dsts)

    plt.plot(exp_data_JPR)
    plt.xlabel("Total Cost")
    plt.show()

if __name__ == "__main__":
    main()