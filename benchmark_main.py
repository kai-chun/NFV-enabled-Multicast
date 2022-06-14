import copy
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Graph
import experience_setting as exp_set
import grouping
import grouping_noQ
import grouping_noPos
import benchmark_unicast
import benchmark_JPR
import benchmark_firstFit
import main_multicast8
import main_multicast9
import main_multicast10
import main_multicast11
import main_multicast12


'''
# 調整各種參數（拓墣、節點數[20-100,200]、使用者比例），印出數據
topology: Agis(25), Dfn(58), Ion(125), Colt(153)

---

'''

def main(topology, bandwidth, exp_num, group, gen_new_data):
    ### Parameters
    alpha = 0.6
    beta = 1 - alpha

    dst_ratio = 0.4
    edge_prob = 0.2

    exp_factor = dst_ratio
    add_factor = 0.1
    bound_factor = 0.4
    factor = topology

    # cost weight: transimission, processing, placing
    weight = (0.6, 0.4, 1)
   
    #vnf_num = random.randint(1,1)
    vnf_type = {'t':1}
    quality_list = {'360p': 0.3, '480p': 0.5, '720p': 0.7, '1080p': 1}
    video_type = [q for q in quality_list]
    user_limit = 1
    is_group = group # If is_group = 0, execute k_means with k = 1.

    ### [0:total cost, 1:transmission cost, 2:processing cost, 3:placing cost, 
    ### 4:avg cost, 5:transcoder num, 6:delay(max path len), 7:running time]

    times = []
    group_num = []

    while exp_factor <= bound_factor: 
        if exp_factor == 0.3: 
            exp_factor += add_factor
            continue
        times.append(exp_factor)
        print('--- ', exp_factor, ' ---')

        ### [0:total cost, 1:transmission cost, 2:processing cost, 3:placing cost, 
        ### 4:avg cost, 5:transcoder num, 6:delay(max path len), 7:running time, 8: failed num]
        exp_data_unicast = [[],[],[],[],[],[],[],[],[]]
        exp_data_JPR = [[],[],[],[],[],[],[],[],[]]
        exp_data_JPR_notReuse = [[],[],[],[],[],[],[],[],[]]
        exp_data_firstFit = [[],[],[],[],[],[],[],[],[]]
        exp_data_main = [[],[],[],[],[],[],[],[],[]]
        exp_data_merge = [[],[],[],[],[],[],[],[],[]]

        ### Execute Algorithms
        if "ER" == factor[:2]:
            node_num = int(factor[3:])
        else:
            input_G = nx.read_gml("topology/"+factor+".gml") # Agis(25), Dfn(58), Ion(125), Colt(153)
            node_num = len(input_G.nodes())

        dst_num = int(np.ceil(node_num * exp_factor))   
        graph_exp = {'bandwidth': bandwidth, 'edge_prob': edge_prob}
        service_exp = {'dst_num': dst_num, 'video_type': video_type}
        
        if gen_new_data == 1:
            for i in range(exp_num):
                exp_set.generate_exp(graph_exp, service_exp, i, factor, exp_factor)

            print('exp ok')

        ### degree
        count_d = dict()
        degree_list = []
        exp_id = []

        for i in range(exp_num):
            if i % 10 == 0:
                print(i, end=' ')
            
            # Read Graph
            input_graph = exp_set.read_exp_graph(graph_exp, i, factor, exp_factor)
            G_unicast = copy.deepcopy(input_graph[0]) 
            G_JPR = copy.deepcopy(input_graph[0]) 
            G_JPR_notReuse = copy.deepcopy(input_graph[0]) 
            G_firstFit = copy.deepcopy(input_graph[0]) 
            G_main = copy.deepcopy(input_graph[0]) 
            G_original = copy.deepcopy(input_graph[0]) 
            pos = input_graph[1]
        
            input_service = exp_set.read_exp_service(graph_exp, i, factor, exp_factor)
            src = input_service[0]
            dsts = input_service[1]

            ### degree
            degree_src = input_graph[0].degree[src]
            
            # if degree_src not in count_d:
            #     count_d[degree_src] = 0

            # count_d[degree_src] += 1
            
            # if count_d[degree_src] > 3: continue

            
            degree_list.append(degree_src)
            # exp_id.append(i)

            # Grouping
            group_list = grouping.k_means(input_graph[0], dsts, video_type, user_limit, is_group)
            # group_list = grouping_noPos.k_means(input_graph[1], dsts, video_type, user_limit, is_group)
            group_num.append(len(group_list))
            # print('group ok')

            G_main_all = list()

            delay_unicast = list()
            delay_JPR = [0]
            delay_JPR_notReuse = [0]
            # delay_JPR = list()
            # delay_JPR_notReuse = list()
            delay_firstFit = list()
            delay_main = list()

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

            running_time_unicast = 0
            running_time_JPR = 0
            running_time_JPR_notReuse = 0
            running_time_firstFit = 0
            running_time_main = 0
            running_time_merge = 0

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
                # G_JPR_ans = benchmark_JPR.search_multipath(G_JPR, service_JPR, alpha, vnf_type, quality_list, isReuse=True)
                running_time_JPR += time.time() - pre_time
                # print('JPR', end=' ')

                pre_time = time.time()
                # G_JPR_notReuse_ans = benchmark_JPR.search_multipath(G_JPR_notReuse, service_JPR_notReuse, alpha, vnf_type, quality_list, isReuse=False)
                running_time_JPR_notReuse += time.time() - pre_time
                # print('JPR_notReuse', end=' ')

                pre_time = time.time()
                G_firstFit_ans = benchmark_firstFit.search_multipath(G_firstFit, pos, service_firstFit, quality_list)
                running_time_firstFit += time.time() - pre_time
                # print('firstFit', end=' ')

                pre_time = time.time()
                G_main_ans = main_multicast12.search_multipath(G_main, service_main, quality_list)
                running_time_main += time.time() - pre_time
                running_time_merge += running_time_main
                # print('main', end=' ')
                G_main_all.append(G_main_ans)

                G_unicast = copy.deepcopy(G_unicast_ans[0])
                # G_JPR = copy.deepcopy(G_JPR_ans[0])
                # G_JPR_notReuse = copy.deepcopy(G_JPR_notReuse_ans[0])
                G_firstFit = copy.deepcopy(G_firstFit_ans[0])
                G_main = copy.deepcopy(G_main_ans[0])

                failed_num_unicast += len(G_unicast_ans[2])
                # if nx.classes.function.is_empty(G_JPR_ans[1]):
                #     failed_num_JPR += len(group)
                # if nx.classes.function.is_empty(G_JPR_notReuse_ans[1]):
                #     failed_num_JPR_notReuse += len(group)
                if nx.classes.function.is_empty(G_firstFit_ans[1]):
                    failed_num_firstFit+= len(group)
                failed_num_main += len(G_main_ans[6])

                transcoder_num_unicast += Graph.count_vnf(G_unicast_ans[-2])
                # transcoder_num_JPR += Graph.count_vnf(G_JPR_ans[-2])
                # transcoder_num_JPR_notReuse += Graph.count_vnf(G_JPR_notReuse_ans[-2])
                transcoder_num_firstFit += Graph.count_vnf(G_firstFit_ans[-2])
                transcoder_num_main += Graph.count_vnf(G_main_ans[-1])

                delay_unicast.append(Graph.max_len(G_unicast_ans[-1]))
                # delay_JPR.append(Graph.max_len(G_JPR_ans[-1]))
                # delay_JPR_notReuse.append(Graph.max_len(G_JPR_notReuse_ans[-1]))
                delay_firstFit.append(Graph.max_len(G_firstFit_ans[-1]))
                delay_main.append(Graph.max_len(G_main_ans[2]))
                
            ### Running merge method
            pre_time = time.time()
            G_merge_ans = main_multicast12.merge_group(G_original, G_main, src, quality_list, G_main_all, weight)
            running_time_merge += time.time() - pre_time
            # print('merge', end=' ')
            
            failed_num_merge = len(G_merge_ans[1])
            transcoder_num_merge = G_merge_ans[-2]

            cost_tuple_unicast = Graph.cal_total_cost(G_unicast, weight, False)
            cost_tuple_JPR = Graph.cal_total_cost(G_JPR, weight, True)
            cost_tuple_notReuse = Graph.cal_total_cost(G_JPR_notReuse, weight, False)
            cost_tuple_firstFit = Graph.cal_total_cost(G_firstFit, weight, False)
            cost_tuple_main = Graph.cal_total_cost(G_main, weight, True)
            cost_tuple_merge = Graph.cal_total_cost(G_merge_ans[0], weight, True)

            ### Total cost, Trans_cost, Proc_cost, Plac_cost (avg to exp times)
            for j in range(4):
                exp_data_unicast[j].append(round(cost_tuple_unicast[j], 2))
                exp_data_JPR[j].append(round(cost_tuple_JPR[j], 2))
                exp_data_JPR_notReuse[j].append(round(cost_tuple_notReuse[j], 2))
                exp_data_firstFit[j].append(round(cost_tuple_firstFit[j], 2))
                exp_data_main[j].append(round(cost_tuple_main[j], 2))
                exp_data_merge[j].append(round(cost_tuple_merge[j], 2))

            ###  Avg cost (avg to user nums)
            total_dst = dst_num
            if total_dst == failed_num_unicast:
                exp_data_unicast[4].append(0)
            else:
                dst_num_unicast = total_dst - failed_num_unicast
                exp_data_unicast[4].append(cost_tuple_unicast[0]/dst_num_unicast)
            if total_dst == failed_num_JPR:
                exp_data_JPR[4].append(0)
            else:
                dst_num_JPR = total_dst - failed_num_JPR
                exp_data_JPR[4].append(cost_tuple_JPR[0]/dst_num_JPR)
            if total_dst == failed_num_JPR_notReuse:
                exp_data_JPR_notReuse[4].append(0)
            else:
                dst_num_JPR_notReuse = total_dst - failed_num_JPR_notReuse
                exp_data_JPR_notReuse[4].append(cost_tuple_notReuse[0]/dst_num_JPR_notReuse)
            if total_dst == failed_num_firstFit:
                exp_data_firstFit[4].append(0)
            else:
                dst_num_firstFit = total_dst - failed_num_firstFit
                exp_data_firstFit[4].append(cost_tuple_firstFit[0]/dst_num_firstFit)
            if total_dst == failed_num_main:
                exp_data_main[4].append(0)
            else:
                dst_num_main = total_dst - failed_num_main
                exp_data_main[4].append(cost_tuple_main[0]/dst_num_main)
            if total_dst == failed_num_merge:
                exp_data_merge[4].append(0)
            else:
                dst_num_merge = total_dst - failed_num_merge
                exp_data_merge[4].append(cost_tuple_merge[0]/dst_num_merge)

            ### Transcoder Number (avg to exp times)
            exp_data_unicast[5].append(transcoder_num_unicast)
            exp_data_JPR[5].append(transcoder_num_JPR)
            exp_data_JPR_notReuse[5].append(transcoder_num_JPR_notReuse)
            exp_data_firstFit[5].append(transcoder_num_firstFit)
            exp_data_main[5].append(transcoder_num_main)
            exp_data_merge[5].append(transcoder_num_merge)

            ### Delay (avg to exp times)
            exp_data_unicast[6].append(max(delay_unicast))
            exp_data_JPR[6].append(max(delay_JPR))
            exp_data_JPR_notReuse[6].append(max(delay_JPR_notReuse))
            exp_data_firstFit[6].append(max(delay_firstFit))
            exp_data_main[6].append(max(delay_main))
            exp_data_merge[6].append(G_merge_ans[-1])

            ### Running time (avg to exp times)
            exp_data_unicast[7].append(running_time_unicast)
            exp_data_JPR[7].append(running_time_JPR)
            exp_data_JPR_notReuse[7].append(running_time_JPR_notReuse)
            exp_data_firstFit[7].append(running_time_firstFit)
            exp_data_main[7].append(running_time_main)
            exp_data_merge[7].append(running_time_merge)

            ### Failed dst ratio (avg to user nums)
            exp_data_unicast[8].append(failed_num_unicast)
            exp_data_JPR[8].append(failed_num_JPR)
            exp_data_JPR_notReuse[8].append(failed_num_JPR_notReuse)
            exp_data_firstFit[8].append(failed_num_firstFit)
            exp_data_main[8].append(failed_num_main)
            exp_data_merge[8].append(failed_num_merge)

        title = {0: "Total_cost", 1: "Trans_cost", 2: "Proc_cost", 3: "Plac_cost", \
            4: "Avg_cost", 5: "Transcoder_num", 6: "Delay", 7: "Running_time", 8: "Failed_num"}
        
        with pd.ExcelWriter('exp_data/'+factor+'_d'+str(exp_factor)+'_bw'+str(bandwidth)+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx') as writer:  
            for i in range(9):
                data = pd.DataFrame({"Unicast":exp_data_unicast[i],"JPR":exp_data_JPR[i],\
                "JPR_notReuse":exp_data_JPR_notReuse[i],"FirstFit":exp_data_firstFit[i],"Main":exp_data_main[i],"Merge":exp_data_merge[i],"degree": degree_list})
                data.to_excel(writer, sheet_name=title[i], index=True)

        ### degree
        # with pd.ExcelWriter('exp_data/'+factor+'_d'+str(exp_factor)+'_bw'+str(bandwidth)+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx') as writer:  
        #     for i in range(9):
        #         data = pd.DataFrame({"exp_id": exp_id, "degree": degree_list,"Merge":exp_data_merge[i]})
        #         data.to_excel(writer, sheet_name=title[i], index=False)

        exp_factor = round(exp_factor + add_factor, 2)

    print("=== ",factor," (",len(input_graph[0].nodes),"), BW =",bandwidth," ===")
    # print(f'{"exp num:"}{exp_num:<4d}|{"unicast":^10}|{"merge":^10}|{"main":^10}|{"JPR":^10}|{"notReuse":^10}|{"firstFit":^10}')
    # print(f'{"total_cost":12}|{exp_data_unicast[0][-1]:^10.3f}|{exp_data_merge[0][-1]:^10.3f}|{exp_data_main[0][-1]:^10.3f}|{exp_data_JPR[0][-1]:^10.3f}|{exp_data_JPR_notReuse[0][-1]:^10.3f}|{exp_data_firstFit[0][-1]:^10.3f}')
    # print(f'{"failed:"}{total_dst:<5d}|{failed_num_unicast:^10d}|{failed_num_merge:^10d}|{failed_num_main:^10d}|{failed_num_JPR:^10d}|{failed_num_JPR_notReuse:^10d}|{failed_num_firstFit:^10d}')
    # print(f'{"avg_cost":12}|{exp_data_unicast[4][-1]:^10.3f}|{exp_data_merge[4][-1]:^10.3f}|{exp_data_main[4][-1]:^10.3f}|{exp_data_JPR[4][-1]:^10.3f}|{exp_data_JPR_notReuse[4][-1]:^10.3f}|{exp_data_firstFit[4][-1]:^10.3f}')
    print(f'{"running time":12}|{exp_data_unicast[7][-1]:^10.3f}|{exp_data_merge[7][-1]:^10.3f}|{exp_data_main[7][-1]:^10.3f}|{exp_data_JPR[7][-1]:^10.3f}|{exp_data_JPR_notReuse[7][-1]:^10.3f}|{exp_data_firstFit[7][-1]:^10.3f}')
    print(group_num)
    print()

# if __name__ == "__main__":
#     main()