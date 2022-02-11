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
# 調整各種參數（拓墣、節點數[20-100,200]、使用者比例），印出數據
topology: Abilene(11), Agis(25), Bellcanada(48), Columbus(71), Oteglobe(93), 
Deltacom(113), Pern(127), GtsCe(149), Cogentco(197)

1. 固定一個可用拓墣（grid, real world）
2. 將成本拆開印出來，並互相對照比較（執行20次）
3. 調整成本，讓曲線趨於平緩，並符合邏輯
4. 調整後，再次執行 50, 100, 1000次

---

1. 將在同個點上執行的 processing 成本忽略 (?)
'''

def main():
    ### Parameters
    alpha = 0.6
    beta = 1 - alpha

    node_num = 25
    dst_ratio = 0.2
    edge_prob = 0.8

    exp_factor = dst_ratio
    add_factor = 0.1
    bound_factor = 0.6
   
    dst_num = int(np.ceil(node_num * dst_ratio))
    vnf_num = random.randint(1,1)
    vnf_type = {'t':1}#,'b':1,'c':1,'d':1,'e':1}
    quality_list = {'360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
    video_type = [q for q in quality_list]
    user_limit = 1
    is_group = 1 # If is_group = 0, execute k_means with k = 1.
    exp_num = 50

    exp_data_unicast = []
    exp_data_JPR = [[],[],[],[]]
    exp_data_JPR_notReuse = [[],[],[],[]]
    exp_data_firstFit = [[],[],[],[]]
    failed_dsts_JPR = []
    times = []
    group_num = []
    factor = 'dst_num'

    #failed_dsts_file = open('exp_img/failed_dsts'+str(dst_ratio)+'.txt', 'w')
    #failed_dsts_file.write("dst_ratio = "+str(dst_ratio)+"\n")

    while exp_factor <= bound_factor: 
        times.append(exp_factor)
        print('--- ', exp_factor, ' ---')

        total_cost_JPR = [0] * 4
        total_cost_JPR_notReuse = [0] * 4
        total_cost_firstFit = [0] * 4
        failed_dsts = 0

        ### Execute Algorithms
        dst_num = int(np.ceil(node_num * exp_factor))   
        graph_exp = {'node_num': node_num, 'edge_prob': edge_prob}
        service_exp = {'dst_num': dst_num, 'video_type': video_type}
        
        # for i in range(exp_num):
        #    exp_set.generate_exp(graph_exp, service_exp, i, factor, exp_factor)

        print('exp ok')

        for i in range(exp_num):
            if i % 10 == 0:
                print(i, end=' ')

            # Read Graph
            input_graph = exp_set.read_exp_graph(i, factor, exp_factor)
            #input_graph = exp_set.read_exp_graph(0, factor, 0.6)
            G_JPR = copy.deepcopy(input_graph[0]) 
            G_JPR_notReuse = copy.deepcopy(input_graph[0]) 
            G_firstFit = copy.deepcopy(input_graph[0]) 
            pos = input_graph[1]
        
            input_service = exp_set.read_exp_service(i, factor, exp_factor)
            #input_service = exp_set.read_exp_service(0, factor, 0.6)
            src = input_service[0]
            dsts = input_service[1]

            # Grouping
            group_list = grouping.k_means(pos, dsts, video_type, user_limit, is_group)
            #group_num.append()
            #print('group ok',len(group_list))

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
                service_JPR_notReuse = copy.deepcopy(service_JPR)
                service_firstFit = copy.deepcopy(service_JPR)
                
                G_JPR_ans = benchmark_JPR.search_multipath(G_JPR, service_JPR, alpha, vnf_type, quality_list, isReuse=True)
                #print('G_JPR', end=' ')
                G_JPR_notReuse_ans = benchmark_JPR.search_multipath(G_JPR_notReuse, service_JPR_notReuse, alpha, vnf_type, quality_list, isReuse=False)
                #print('G_JPR_notReuse', end=' ')
                G_firstFit_ans = benchmark_firstFit.search_multipath(G_firstFit, pos, service_firstFit, quality_list)
                #print('firstFit', end=' ')

                G_JPR = copy.deepcopy(G_JPR_ans[0])
                G_JPR_notReuse = copy.deepcopy(G_JPR_notReuse_ans[0])
                G_firstFit = copy.deepcopy(G_firstFit_ans[0])

                # if nx.classes.function.is_empty(G_JPR_ans[1]):
                #     failed_dsts += len(group)
                # print(Graph.cal_trans_cost(G_firstFit_ans[1]))
                # print("group", i," : ", Graph.cal_total_cost(G_firstFit)) 
            
            cost_tuple_JPR = Graph.cal_total_cost(G_JPR)
            cost_tuple_notReuse = Graph.cal_total_cost(G_JPR_notReuse)
            cost_tuple_firstFit = Graph.cal_total_cost(G_firstFit)
            
            
            for j in range(4):
                total_cost_JPR[j] = round(total_cost_JPR[j] + cost_tuple_JPR[j], 2)
                total_cost_JPR_notReuse[j] = round(total_cost_JPR_notReuse[j] + cost_tuple_notReuse[j], 2)
                total_cost_firstFit[j] = round(total_cost_firstFit[j] + cost_tuple_firstFit[j], 2)
        
        print(' ')
        for j in range(4):
            exp_data_JPR[j].append(total_cost_JPR[j]/exp_num)
            exp_data_JPR_notReuse[j].append(total_cost_JPR_notReuse[j]/exp_num)
            exp_data_firstFit[j].append(total_cost_firstFit[j]/exp_num)

        #failed_dsts_file.write("node_num = "+str(node_num)+", failed_dsts = "+str(failed_dsts)+"\n")
        exp_factor = round(exp_factor + add_factor, 2)
        print("total_cost = ", round(total_cost_JPR[0]/exp_num,3), " / ", round(total_cost_JPR_notReuse[0]/exp_num,3), " / ", round(total_cost_firstFit[0]/exp_num,3))
    
    #failed_dsts_file.close()

    fig,ax = plt.subplots(4,1)
    if is_group == 1:
        fig.suptitle('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n') 
    else:
        fig.suptitle('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1') 

    ax[0].plot(times, exp_data_JPR[0])
    ax[0].plot(times, exp_data_JPR_notReuse[0])
    ax[0].plot(times, exp_data_firstFit[0], '--')
    ax[0].set_xlabel("Dst Ratio")
    ax[0].set_ylabel("Total Cost")
    
    ax[1].plot(times, exp_data_JPR[1])
    ax[1].plot(times, exp_data_JPR_notReuse[1])
    ax[1].plot(times, exp_data_firstFit[1], '--')
    ax[1].set_xlabel("Dst Ratio")
    ax[1].set_ylabel("Transmission")
    
    ax[2].plot(times, exp_data_JPR[2])
    ax[2].plot(times, exp_data_JPR_notReuse[2])
    ax[2].plot(times, exp_data_firstFit[2], '--')
    ax[2].set_xlabel("Dst Ratio")
    ax[2].set_ylabel("Processing")
    
    ax[3].plot(times, exp_data_JPR[3])
    ax[3].plot(times, exp_data_JPR_notReuse[3])
    ax[3].plot(times, exp_data_firstFit[3], '--')
    ax[3].set_xlabel("Dst Ratio")
    ax[3].set_ylabel("Placing")
    
    #plt.text("")
    plt.legend(['JPR','JPR_notReue','FirstFit'], bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()

    if is_group == 1:
        plt.savefig('exp_img/all_cost_'+str(exp_num)+'_k.png')
    else:
        plt.savefig('exp_img/all_cost_'+str(exp_num)+'.png')
    
    plt.close()

    plt.plot(times, exp_data_JPR[0])
    plt.plot(times, exp_data_JPR_notReuse[0])
    plt.plot(times, exp_data_firstFit[0], '--')
    plt.legend(['JPR','JPR_notReue','FirstFit'], loc="lower right")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/total_cost_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/total_cost_'+str(exp_num)+'.png')
    
    plt.close()
    

if __name__ == "__main__":
    main()