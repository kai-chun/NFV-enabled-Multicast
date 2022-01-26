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
2. 調整各種參數（拓墣、節點數[20-100,200]、使用者比例），印出數據
topology: Abilene(11), Agis(25), Bellcanada(48), Columbus(71), Oteglobe(93), 
Deltacom(113), Pern(127), GtsCe(149), Cogentco(197)
'''

def main():
    ### Parameters
    alpha = 0.6
    beta = 1 - alpha

    node_num = 11
    dst_ratio = 0.2
    edge_prob = 0.8

    exp_factor = dst_ratio
    add_factor = 0.1
    bound_factor = 0.2
   
    dst_num = int(np.ceil(node_num * dst_ratio))
    vnf_num = random.randint(1,1)
    vnf_type = {'t':1}#,'b':1,'c':1,'d':1,'e':1}
    quality_list = {'360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
    video_type = [q for q in quality_list]
    user_limit = 1
    exp_num = 1

    ### Execute Algorithms
    graph_exp = {'node_num': node_num, 'edge_prob': edge_prob}
    service_exp = {'dst_num': dst_num, 'video_type': video_type}

    exp_data_unicast = []
    exp_data_JPR = []
    exp_data_firstFit = []
    failed_dsts_JPR = []
    times = []
    factor = 'node_num'

    failed_dsts_file = open('exp_img/failed_dsts'+str(dst_ratio)+'.txt', 'w')
    failed_dsts_file.write("dst_ratio = "+str(dst_ratio)+"\n")

    while exp_factor <= bound_factor: 
        times.append(exp_factor)
        print('--- ', exp_factor, ' ---')

        total_cost_JPR = 0
        total_cost_firstFit = 0
        failed_dsts = 0

        for i in range(exp_num):
            exp_set.generate_exp(graph_exp, service_exp, i, factor, node_num)

        print('exp ok')

        for i in range(exp_num):
            if i % 100 == 0:
                print(i, end=' ')

            # Read Graph
            input_graph = exp_set.read_exp_graph(i, factor, node_num)
            G_JPR = copy.deepcopy(input_graph[0]) 
            G_firstFit = copy.deepcopy(input_graph[0]) 
            pos = input_graph[1]
        
            input_service = exp_set.read_exp_service(i, factor, node_num)
            src = input_service[0]
            dsts = input_service[1]

            # Grouping
            group_list = grouping.k_means(pos, dsts, video_type, user_limit)
            #print('group ok')

            # Find multicast path for all groups
            for i in group_list:
                group = group_list[i]
                group_info = dict()
                sfc = dict()
                for g in group:
                    group_info[g] = dsts[g]
                    sfc[g] = []
                
                best_quality = video_type[max(video_type.index(group_info[g]) for g in group_info)]
                service_JPR = (src, group_info, sfc, best_quality)
                service_firstFit = copy.deepcopy(service_JPR)
                G_JPR_ans = benchmark_JPR.search_multipath(G_JPR, service_JPR, alpha, vnf_type, quality_list)
                #print('JPR ok')
                G_firstFit_ans = benchmark_firstFit.search_multipath(G_firstFit, pos, service_firstFit, quality_list)
                #print('firstFit ok')
                G_JPR = copy.deepcopy(G_JPR_ans[0])
                G_firstFit = copy.deepcopy(G_firstFit_ans[0])

                if nx.classes.function.is_empty(G_JPR_ans[1]):
                    failed_dsts += len(group)
            
            total_cost_JPR += Graph.cal_total_cost(G_JPR)
            total_cost_firstFit += Graph.cal_total_cost(G_firstFit)
         
        print(' ')
        exp_data_JPR.append(total_cost_JPR/exp_num)
        exp_data_firstFit.append(total_cost_firstFit/exp_num)
        failed_dsts_file.write("node_num = "+str(node_num)+", failed_dsts = "+str(failed_dsts)+"\n")
        exp_factor += add_factor
        print("total_cost = ", total_cost_JPR, " / ", total_cost_firstFit)
        #print("failed dsts = ", failed_dsts)
    
    failed_dsts_file.close()

    plt.title('Abilene ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+')') 
    plt.xlabel("Node Numer")
    plt.ylabel("Total Cost")
    plt.plot(times, exp_data_JPR)
    plt.plot(times, exp_data_firstFit, '--')
    plt.legend(['JPR','FirstFit'], loc='lower right')
    plt.savefig('exp_img/Abilene.png')
    plt.close()

if __name__ == "__main__":
    main()