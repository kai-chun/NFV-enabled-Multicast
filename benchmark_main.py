import copy
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import Graph
import experience_setting as exp_set
import grouping
import benchmark_unicast
import benchmark_JPR
import benchmark_firstFit
import main_multicast6


'''
# 調整各種參數（拓墣、節點數[20-100,200]、使用者比例），印出數據
topology: Agis(25), Dfn(58), Ion(125), Colt(153)

---

'''

def main():
    ### Parameters
    alpha = 0.6
    beta = 1 - alpha

    dst_ratio = 0.2
    edge_prob = 0.8 

    exp_factor = dst_ratio
    add_factor = 0.1
    bound_factor = 0.6
    factor = 'Ion'

    # cost weight: transimission, processing, placing
    weight = (0.6, 0.4, 1)
   
    #vnf_num = random.randint(1,1)
    vnf_type = {'t':1}
    quality_list = {'360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
    video_type = [q for q in quality_list]
    user_limit = 1
    is_group = 1 # If is_group = 0, execute k_means with k = 1.
    exp_num = 50

    ### [0:total cost, 1:transmission cost, 2:processing cost, 3:placing cost, 
    ### 4:avg cost, 5:transcoder num, 6:delay(max path len), 7:running time]
    exp_data_unicast = [[],[],[],[],[],[],[],[]]
    exp_data_JPR = [[],[],[],[],[],[],[],[]]
    exp_data_JPR_notReuse = [[],[],[],[],[],[],[],[]]
    exp_data_firstFit = [[],[],[],[],[],[],[],[]]
    exp_data_main = [[],[],[],[],[],[],[],[]]
    exp_data_merge = [[],[],[],[],[],[],[],[]]

    failed_data_unicast = []
    failed_data_JPR = []
    failed_data_JPR_notReuse = []
    failed_data_firstFit = []
    failed_data_main = []
    failed_data_merge = []

    times = []
    group_num = []

    while exp_factor <= bound_factor: 
        times.append(exp_factor)
        print('--- ', exp_factor, ' ---')

        total_cost_unicast = [0] * 4
        total_cost_JPR = [0] * 4
        total_cost_JPR_notReuse = [0] * 4
        total_cost_firstFit = [0] * 4
        total_cost_main = [0] * 4
        total_cost_merge = [0] * 4
        
        failed_num_unicast = 0
        failed_num_JPR = 0
        failed_num_JPR_notReuse = 0
        failed_num_firstFit = 0
        failed_num_main = 0
        failed_num_merge = 0

        transcoder_num_unicast = 0
        transcoder_num_JPR = 0
        transcoder_num_JPR_notReuse = 0
        transcoder_num_firstFit = 0
        transcoder_num_main = 0
        transcoder_num_merge = 0

        total_delay_unicast = 0
        total_delay_JPR = 0
        total_delay_JPR_notReuse = 0
        total_delay_firstFit = 0
        total_delay_main = 0
        total_delay_merge = 0

        running_time_unicast = 0
        running_time_JPR = 0
        running_time_JPR_notReuse = 0
        running_time_firstFit = 0
        running_time_main = 0
        running_time_merge = 0

        ### Execute Algorithms
        input_G = nx.read_gml("topology/"+factor+".gml") # Agis(25), Dfn(58), Ion(125), Colt(153)
        node_num = len(input_G.nodes())

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
            G_unicast = copy.deepcopy(input_graph[0]) 
            G_JPR = copy.deepcopy(input_graph[0]) 
            G_JPR_notReuse = copy.deepcopy(input_graph[0]) 
            G_firstFit = copy.deepcopy(input_graph[0]) 
            G_main = copy.deepcopy(input_graph[0]) 
            G_original = copy.deepcopy(input_graph[0]) 
            pos = input_graph[1]
        
            input_service = exp_set.read_exp_service(i, factor, exp_factor)
            src = input_service[0]
            dsts = input_service[1]

            # Grouping
            group_list = grouping.k_means(pos, dsts, video_type, user_limit, is_group)
            #group_num.append()
            # print('group ok',len(group_list))

            G_main_all = list()

            delay_unicast = 0
            delay_JPR = 0
            delay_JPR_notReuse = 0
            delay_firstFit = 0
            delay_main = 0

            # Find multicast path for all groups
            for j in group_list:
                group = group_list[j]
                group_info = dict()
                sfc = dict()
                for g in group:
                    group_info[g] = dsts[g]
                    sfc[g] = []
                
                best_quality = video_type[max(video_type.index(group_info[g]) for g in group_info)]
                service_unicast = (src, group_info, sfc, best_quality)
                service_JPR = copy.deepcopy(service_unicast)
                service_JPR_notReuse = copy.deepcopy(service_unicast)
                service_firstFit = copy.deepcopy(service_unicast)
                service_main = copy.deepcopy(service_unicast)
                
                ### Running methods
                pre_time = time.time()
                G_unicast_ans = benchmark_unicast.search_unicast_path(G_unicast, pos, service_unicast, quality_list)
                running_time_unicast += time.time() - pre_time
                # print('unicast', end=' ')

                pre_time = time.time()
                G_JPR_ans = benchmark_JPR.search_multipath(G_JPR, service_JPR, alpha, vnf_type, quality_list, isReuse=True)
                running_time_JPR += time.time() - pre_time
                # print('JPR', end=' ')

                pre_time = time.time()
                G_JPR_notReuse_ans = benchmark_JPR.search_multipath(G_JPR_notReuse, service_JPR_notReuse, alpha, vnf_type, quality_list, isReuse=False)
                running_time_JPR_notReuse += time.time() - pre_time
                # print('JPR_notReuse', end=' ')

                pre_time = time.time()
                G_firstFit_ans = benchmark_firstFit.search_multipath(G_firstFit, pos, service_firstFit, quality_list)
                running_time_firstFit += time.time() - pre_time
                # print('firstFit', end=' ')

                pre_time = time.time()
                G_main_ans = main_multicast6.search_multipath(G_main, service_main, quality_list)
                running_time_main += time.time() - pre_time
                running_time_merge += running_time_main
                # print('main', end=' ')
                G_main_all.append(G_main_ans)

                G_unicast = copy.deepcopy(G_unicast_ans[0])
                G_JPR = copy.deepcopy(G_JPR_ans[0])
                G_JPR_notReuse = copy.deepcopy(G_JPR_notReuse_ans[0])
                G_firstFit = copy.deepcopy(G_firstFit_ans[0])
                G_main = copy.deepcopy(G_main_ans[0])

                failed_num_unicast += len(G_unicast_ans[2])
                if nx.classes.function.is_empty(G_JPR_ans[1]):
                    failed_num_JPR += len(group)
                if nx.classes.function.is_empty(G_JPR_notReuse_ans[1]):
                    failed_num_JPR_notReuse += len(group)
                if nx.classes.function.is_empty(G_firstFit_ans[1]):
                    failed_num_firstFit+= len(group)
                failed_num_main += len(G_main_ans[6])

                transcoder_num_unicast += Graph.count_vnf(G_unicast_ans[-2])
                transcoder_num_JPR += Graph.count_vnf(G_JPR_ans[-2])
                transcoder_num_JPR_notReuse += Graph.count_vnf(G_JPR_notReuse_ans[-2])
                transcoder_num_firstFit += Graph.count_vnf(G_firstFit_ans[-2])
                transcoder_num_main += Graph.count_vnf(G_main_ans[-1])

                delay_unicast += Graph.max_len(G_unicast_ans[-1])
                delay_JPR += Graph.max_len(G_JPR_ans[-1])
                delay_JPR_notReuse += Graph.max_len(G_JPR_notReuse_ans[-1])
                delay_firstFit += Graph.max_len(G_firstFit_ans[-1])
                delay_main += Graph.max_len(G_main_ans[2])
                
            ### Running merge method
            pre_time = time.time()
            G_merge_ans = main_multicast6.merge_group(G_original, G_main, src, quality_list, G_main_all, weight)
            running_time_merge += time.time() - pre_time
            # print('merge', end=' ')
            
            failed_num_merge += len(G_merge_ans[1])
            transcoder_num_merge += G_merge_ans[-2]

            cost_tuple_unicast = Graph.cal_total_cost(G_unicast, weight, False)
            cost_tuple_JPR = Graph.cal_total_cost(G_JPR, weight, True)
            cost_tuple_notReuse = Graph.cal_total_cost(G_JPR_notReuse, weight, False)
            cost_tuple_firstFit = Graph.cal_total_cost(G_firstFit, weight, False)
            cost_tuple_main = Graph.cal_total_cost(G_main, weight, True)
            cost_tuple_merge = Graph.cal_total_cost(G_merge_ans[0], weight, True)
            
            for j in range(4):
                total_cost_unicast[j] = round(total_cost_unicast[j] + cost_tuple_unicast[j], 2)
                total_cost_JPR[j] = round(total_cost_JPR[j] + cost_tuple_JPR[j], 2)
                total_cost_JPR_notReuse[j] = round(total_cost_JPR_notReuse[j] + cost_tuple_notReuse[j], 2)
                total_cost_firstFit[j] = round(total_cost_firstFit[j] + cost_tuple_firstFit[j], 2)
                total_cost_main[j] = round(total_cost_main[j] + cost_tuple_main[j], 2)
                total_cost_merge[j] = round(total_cost_merge[j] + cost_tuple_merge[j], 2)
            
            total_delay_unicast += delay_unicast / len(group_list)
            total_delay_JPR += delay_JPR / len(group_list)
            total_delay_JPR_notReuse += delay_JPR_notReuse / len(group_list)
            total_delay_firstFit += delay_firstFit / len(group_list)
            total_delay_main += delay_main / len(group_list)
            total_delay_merge += G_merge_ans[-1]
        
        print(' ')
        for j in range(4):
            exp_data_unicast[j].append(total_cost_unicast[j]/exp_num)
            exp_data_JPR[j].append(total_cost_JPR[j]/exp_num)
            exp_data_JPR_notReuse[j].append(total_cost_JPR_notReuse[j]/exp_num)
            exp_data_firstFit[j].append(total_cost_firstFit[j]/exp_num)
            exp_data_main[j].append(total_cost_main[j]/exp_num)
            exp_data_merge[j].append(total_cost_merge[j]/exp_num)

        total_dst = dst_num * exp_num
        if total_dst == failed_num_unicast:
            exp_data_unicast[4].append(total_cost_unicast[0])
        else:
            exp_data_unicast[4].append(total_cost_unicast[0]/(total_dst-failed_num_unicast))
        if total_dst == failed_num_JPR:
            exp_data_JPR[4].append(total_cost_JPR[0])
        else:
            exp_data_JPR[4].append(total_cost_JPR[0]/(total_dst-failed_num_JPR))
        if total_dst == failed_num_JPR_notReuse:
            exp_data_JPR_notReuse[4].append(total_cost_JPR_notReuse[0])
        else:
            exp_data_JPR_notReuse[4].append(total_cost_JPR_notReuse[0]/(total_dst-failed_num_JPR_notReuse))
        if total_dst == failed_num_firstFit:
            exp_data_firstFit[4].append(total_cost_firstFit[0])
        else:
            exp_data_firstFit[4].append(total_cost_firstFit[0]/(total_dst-failed_num_firstFit))
        if total_dst == failed_num_main:
            exp_data_main[4].append(total_cost_main[0])
        else:
            exp_data_main[4].append(total_cost_main[0]/(total_dst-failed_num_main))
        if total_dst == failed_num_merge:
            exp_data_merge[4].append(total_cost_merge[0])
        else:
            exp_data_merge[4].append(total_cost_merge[0]/(total_dst-failed_num_merge))

        ### Failed dst ratio
        failed_data_unicast.append((failed_num_unicast/total_dst)/exp_num)
        failed_data_JPR.append((failed_num_JPR/total_dst)/exp_num)
        failed_data_JPR_notReuse.append((failed_num_JPR_notReuse/total_dst)/exp_num)
        failed_data_firstFit.append((failed_num_firstFit/total_dst)/exp_num)
        failed_data_main.append((failed_num_main/total_dst)/exp_num)
        failed_data_merge.append((failed_num_merge/total_dst)/exp_num)

        ### Transcoder Number
        exp_data_unicast[5].append(transcoder_num_unicast/exp_num)
        exp_data_JPR[5].append(transcoder_num_JPR/exp_num)
        exp_data_JPR_notReuse[5].append(transcoder_num_JPR_notReuse/exp_num)
        exp_data_firstFit[5].append(transcoder_num_firstFit/exp_num)
        exp_data_main[5].append(transcoder_num_main/exp_num)
        exp_data_merge[5].append(transcoder_num_merge/exp_num)

        ### Delay
        exp_data_unicast[6].append(total_delay_unicast/exp_num)
        exp_data_JPR[6].append(total_delay_JPR/exp_num)
        exp_data_JPR_notReuse[6].append(total_delay_JPR_notReuse/exp_num)
        exp_data_firstFit[6].append(total_delay_firstFit/exp_num)
        exp_data_main[6].append(total_delay_main/exp_num)
        exp_data_merge[6].append(total_delay_merge/exp_num)

        ### Running time
        exp_data_unicast[7].append(running_time_unicast/exp_num)
        exp_data_JPR[7].append(running_time_JPR/exp_num)
        exp_data_JPR_notReuse[7].append(running_time_JPR_notReuse/exp_num)
        exp_data_firstFit[7].append(running_time_firstFit/exp_num)
        exp_data_main[7].append(running_time_main/exp_num)
        exp_data_merge[7].append(running_time_merge/exp_num)

        exp_factor = round(exp_factor + add_factor, 2)
        print("=== ",factor," (",len(input_graph[0].nodes),")"," ===")
        print(f'{"exp num:"}{exp_num:<4d}|{"unicast":^10}|{"merge":^10}|{"main":^10}|{"JPR":^10}|{"notReuse":^10}|{"firstFit":^10}')
        print(f'{"total_cost":12}|{exp_data_unicast[0][-1]:^10.3f}|{exp_data_merge[0][-1]:^10.3f}|{exp_data_main[0][-1]:^10.3f}|{exp_data_JPR[0][-1]:^10.3f}|{exp_data_JPR_notReuse[0][-1]:^10.3f}|{exp_data_firstFit[0][-1]:^10.3f}')
        print(f'{"failed:"}{total_dst:<5d}|{failed_num_unicast:^10d}|{failed_num_merge:^10d}|{failed_num_main:^10d}|{failed_num_JPR:^10d}|{failed_num_JPR_notReuse:^10d}|{failed_num_firstFit:^10d}')
        print(f'{"avg_cost":12}|{exp_data_unicast[4][-1]:^10.3f}|{exp_data_merge[4][-1]:^10.3f}|{exp_data_main[4][-1]:^10.3f}|{exp_data_JPR[4][-1]:^10.3f}|{exp_data_JPR_notReuse[4][-1]:^10.3f}|{exp_data_firstFit[4][-1]:^10.3f}')
        print(f'{"running time":12}|{exp_data_unicast[7][-1]:^10.3f}|{exp_data_merge[7][-1]:^10.3f}|{exp_data_main[7][-1]:^10.3f}|{exp_data_JPR[7][-1]:^10.3f}|{exp_data_JPR_notReuse[7][-1]:^10.3f}|{exp_data_firstFit[7][-1]:^10.3f}')


    ### Print data: failed num of dst.
    plt.plot(times, failed_data_unicast, ':', color="#2ca02c")
    plt.plot(times, failed_data_JPR, color="#1f77b4")
    plt.plot(times, failed_data_JPR_notReuse, color="#ff7f0e")
    plt.plot(times, failed_data_firstFit, '--', color="#9467bd")
    plt.plot(times, failed_data_main, color="#e377c2")
    plt.plot(times, failed_data_merge, '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Failed Ratio")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/failed_dst_num_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/failed_dst_num_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: each cost.
    fig,ax = plt.subplots(4,1)
    if is_group == 1:
        fig.suptitle('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n') 
    else:
        fig.suptitle('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1') 

    ax[0].plot(times, exp_data_unicast[0], ':', color="#2ca02c")
    ax[0].plot(times, exp_data_JPR[0], color="#1f77b4")
    ax[0].plot(times, exp_data_JPR_notReuse[0], color="#ff7f0e")
    ax[0].plot(times, exp_data_firstFit[0], '--', color="#9467bd")
    ax[0].plot(times, exp_data_main[0], color="#e377c2")
    ax[0].set_xlabel("Dst Ratio")
    ax[0].set_ylabel("Total Cost")
    
    ax[1].plot(times, exp_data_unicast[1], ':', color="#2ca02c")
    ax[1].plot(times, exp_data_JPR[1], color="#1f77b4")
    ax[1].plot(times, exp_data_JPR_notReuse[1], color="#ff7f0e")
    ax[1].plot(times, exp_data_firstFit[1], '--', color="#9467bd")
    ax[1].plot(times, exp_data_main[1], color="#e377c2")
    ax[1].set_xlabel("Dst Ratio")
    ax[1].set_ylabel("Transmission")
    
    ax[2].plot(times, exp_data_unicast[2], ':', color="#2ca02c")
    ax[2].plot(times, exp_data_JPR[2], color="#1f77b4")
    ax[2].plot(times, exp_data_JPR_notReuse[2], color="#ff7f0e")
    ax[2].plot(times, exp_data_firstFit[2], '--', color="#9467bd")
    ax[2].plot(times, exp_data_main[2], color="#e377c2")
    ax[2].set_xlabel("Dst Ratio")
    ax[2].set_ylabel("Processing")
    
    ax[3].plot(times, exp_data_unicast[3], ':', color="#2ca02c")
    ax[3].plot(times, exp_data_JPR[3], color="#1f77b4")
    ax[3].plot(times, exp_data_JPR_notReuse[3], color="#ff7f0e")
    ax[3].plot(times, exp_data_firstFit[3], '--', color="#9467bd")
    ax[3].plot(times, exp_data_main[3], color="#e377c2")
    ax[3].set_xlabel("Dst Ratio")
    ax[3].set_ylabel("Placing")
    
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main'], bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()

    if is_group == 1:
        plt.savefig('exp_img/all_cost_'+str(exp_num)+'_k.png')
    else:
        plt.savefig('exp_img/all_cost_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: total cost.
    plt.plot(times, exp_data_unicast[0], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[0], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[0], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[0], '--', color="#9467bd")
    plt.plot(times, exp_data_main[0], color="#e377c2")
    plt.plot(times, exp_data_merge[0], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Total Cost")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout()
        plt.savefig('exp_img/total_cost_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/total_cost_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: average cost of serviced dst.
    plt.plot(times, exp_data_unicast[4], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[4], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[4], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[4], '--', color="#9467bd")
    plt.plot(times, exp_data_main[4], color="#e377c2")
    plt.plot(times, exp_data_merge[4], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Average Cost")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/avg_cost_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/avg_cost_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: transition cost.
    plt.plot(times, exp_data_unicast[1], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[1], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[1], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[1], '--', color="#9467bd")
    plt.plot(times, exp_data_main[1], color="#e377c2")
    plt.plot(times, exp_data_merge[1], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Transmission Cost")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/tran_cost_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/tran_cost_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: processing cost.
    plt.plot(times, exp_data_unicast[2], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[2], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[2], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[2], '--', color="#9467bd")
    plt.plot(times, exp_data_main[2], color="#e377c2")
    plt.plot(times, exp_data_merge[2], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Processing Cost")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/proc_cost_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/proc_cost_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: placing cost.
    plt.plot(times, exp_data_unicast[3], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[3], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[3], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[3], '--', color="#9467bd")
    plt.plot(times, exp_data_main[3], color="#e377c2")
    plt.plot(times, exp_data_merge[3], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Placing Cost")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/plac_cost_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/plac_cost_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: transcoder num.
    plt.plot(times, exp_data_unicast[5], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[5], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[5], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[5], '--', color="#9467bd")
    plt.plot(times, exp_data_main[5], color="#e377c2")
    plt.plot(times, exp_data_merge[5], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Transcoder Num")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/transcoder_num_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/transcoder_num_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: delay(max path len).
    plt.plot(times, exp_data_unicast[6], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[6], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[6], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[6], '--', color="#9467bd")
    plt.plot(times, exp_data_main[6], color="#e377c2")
    plt.plot(times, exp_data_merge[6], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Delay")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/delay_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/delay_'+str(exp_num)+'.png')
    
    plt.close()

    ### Print data: Running time.
    plt.plot(times, exp_data_unicast[7], ':', color="#2ca02c")
    plt.plot(times, exp_data_JPR[7], color="#1f77b4")
    plt.plot(times, exp_data_JPR_notReuse[7], color="#ff7f0e")
    plt.plot(times, exp_data_firstFit[7], '--', color="#9467bd")
    plt.plot(times, exp_data_main[7], color="#e377c2")
    plt.plot(times, exp_data_merge[7], '--', color="#e377c2")
    plt.legend(['Unicast','JPR','JPR_notReuse','FirstFit','Main','Merge'], loc="lower right")
    plt.xlabel("Dst Ratio")
    plt.ylabel("Running time(s)")
    
    if is_group == 1:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = n')
        plt.tight_layout() 
        plt.savefig('exp_img/running_time_'+str(exp_num)+'_k.png')
    else:
        plt.title('Graph ('+str(len(input_graph[0].nodes))+', '+str(len(input_graph[0].edges))+'), k = 1')
        plt.tight_layout()
        plt.savefig('exp_img/running_time_'+str(exp_num)+'.png')
    
    plt.close()
    
if __name__ == "__main__":
    main()