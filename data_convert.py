import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

title = {0: "Total_cost", 1: "Trans_cost", 2: "Proc_cost", 3: "Plac_cost", \
    4: "Avg_cost", 5: "Transcoder_num", 6: "Delay", 7: "Running_time", 8: "Failed_num"}
              
# topology_list = ["Agis", "Dfn", "Ion", "Colt"]
node_num = {"Agis":25, "Dfn":58, "Ion":125, "Colt":153, "ER_25":25, "ER_50":50, "ER_75":75, "ER_125":125}
node_num_list = [25,58,125,153]
# bandwidth_setting = [2, 5, 10, 20]
# exp_factor = [0.1, 0.2, 0.4]
exp_num = 50
# is_group = 0

# sheet_name = "Total_cost"
# col_name = ["Unicast","JPR","JPR_notReuse","FirstFit","Main_noG","Merge_noG","Main","Merge"]
col_name = ["Main","Merge"]
folder_name = "raw_data0711"

# bw = 20, dst = 0.4 in every topology with g0, g1
def group_data():
    topology_list = ["Agis", "Dfn", "Ion", "Colt"]#["ER_25", "ER_50","ER_75","ER_125"]
    node_num_list = [25,58,125,153]#[25,50,75,125]
    dst_ratio = 0.4
    bandwidth_setting = [20]
    is_group = 1

    sheet_name = title[0]

    exp_data = [[],[],[],[],[],[],[],[]]

    for topology in topology_list:
        file_name = 'exp_data/'+folder_name+'/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        failed_dst_file = pd.read_excel(file_name, sheet_name="Failed_num", usecols=col_name)
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            failed_dst_list = failed_dst_file[col].tolist()
            service_dst = len(exp_list) * node_num[topology] * dst_ratio - sum(failed_dst_list)
            # exp_data[i].append(sum(exp_list)/service_dst)
            exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    # data = pd.DataFrame({"topology":node_num_list,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main_noG":exp_data[4],"Merge_noG":exp_data[5],"Main":exp_data[6],"Merge":exp_data[7]})
    data = pd.DataFrame({"topology":node_num_list,"Main":exp_data[0],"Merge":exp_data[1]})
    data.to_csv(file_path, index=False)

# dst = 0.4, group = 1 in Dfn,Ion with different bandwidth
def bw_failed_data():
    topology_list = ["Dfn","Ion"]#, "Ion"]
    bandwidth_setting = [2, 5, 10, 20]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[8]

    for topology in topology_list:
        exp_data = [[],[],[],[],[],[],[],[]]
        for bw in bandwidth_setting:
            for i,col in enumerate(col_name):
                file_name = 'exp_data/'+folder_name+'/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bw)+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
                input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
            
                exp_list = input_file[col].tolist()
                exp_data[i].append(sum(exp_list)/(len(exp_list)*node_num[topology]*dst_ratio))

        file_path = 'exp_data/bw_fail/'+topology+'_d'+str(dst_ratio)+'_g'+str(is_group)+'.csv'
        data = pd.DataFrame({"bandwidth":bandwidth_setting,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main_noG":exp_data[4],"Merge_noG":exp_data[5],"Main":exp_data[6],"Merge":exp_data[7]})
        data.to_csv(file_path, index=False)


# bw = 20, dst = 0.4, group = 1 in every topology
def place_cost_transocer_num_data():
    topology_list = ["Agis", "Dfn", "Ion","Colt"]
    node_num_list = [25,58,125,153]
    bandwidth_setting = [20]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[3] #3,5

    exp_data = [[],[],[],[],[],[],[],[]]

    for topology in topology_list:
        file_name = 'exp_data/'+folder_name+'/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            if sheet_name == "Plac_cost":
                total_plac = 5 * node_num[topology]
                orig_cost = sum(exp_list)/100*total_plac
                exp_data[i].append(orig_cost/len(exp_list))
            else:
                exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/transcoder_place_cost/'+sheet_name+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'_main.csv'
    # data = pd.DataFrame({"topology":node_num_list,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main_noG":exp_data[4],"Merge_noG":exp_data[5],"Main":exp_data[6],"Merge":exp_data[7]})
    data = pd.DataFrame({"topology":node_num_list,"Main":exp_data[0],"Merge":exp_data[1]})

    data.to_csv(file_path, index=False)

# bw = 20, dst = 0.4, group = 1 in every topology
def all_data_network():
    topology_list = ["Agis", "Dfn", "Ion", "Colt"]
    node_num_list = [25,58,125,153]
    bandwidth_setting = [20]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[3] #0,1,2,3,6,7

    exp_data = [[],[],[],[],[],[],[],[]]

    for topology in topology_list:
        file_name = 'exp_data/'+folder_name+'/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        failed_dst_file = pd.read_excel(file_name, sheet_name="Failed_num", usecols=col_name)

        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()            
            # exp_data[i].append(sum(exp_list)/len(exp_list))
            failed_dst_list = failed_dst_file[col].tolist()
            dst_num = int(np.ceil(node_num[topology] * dst_ratio))
            service_dst = len(exp_list) * dst_num - sum(failed_dst_list)
            # exp_data[i].append(sum(exp_list)/service_dst*10)
            exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/'+sheet_name+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    # data = pd.DataFrame({"topology":node_num_list,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main_noG":exp_data[4],"Merge_noG":exp_data[5],"Main":exp_data[6],"Merge":exp_data[7]})
    data = pd.DataFrame({"topology":node_num_list,"Main":exp_data[0],"Merge":exp_data[1]})
    data.to_csv(file_path, index=False)

def all_data_dst_ratio():
    topology_list = ["Ion"]
    bandwidth_setting = [20]
    dst_ratio = [0.1,0.2,0.4]
    is_group = 1

    sheet_name = title[0] #0,1,2,3,6,7

    exp_data = [[],[],[],[],[],[],[],[]]
    for r in dst_ratio:
        file_name = 'exp_data/'+folder_name+'/'+topology_list[0]+'_d'+str(r)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        failed_dst_file = pd.read_excel(file_name, sheet_name="Failed_num", usecols=col_name)
        
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            # if col == "Total_cost":
            #     failed_dst_list = failed_dst_file[col].tolist()
            #     dst_num = int(np.ceil(node_num[topology_list[0]] * dst_ratio))
            #     service_dst = len(exp_list) * dst_num - sum(failed_dst_list)
            #     exp_data[i].append(sum(exp_list)/service_dst*10)
            # else:
            exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/'+sheet_name+'_'+topology_list[0]+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    # data = pd.DataFrame({"dst_ratio":dst_ratio,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main_noG":exp_data[4],"Merge_noG":exp_data[5],"Main":exp_data[6],"Merge":exp_data[7]})
    data = pd.DataFrame({"dst_ratio":dst_ratio,"Main":exp_data[0],"Merge":exp_data[1]})
    data.to_csv(file_path, index=False)

def compare_self_network():
    topology_list = ["Agis", "Dfn", "Ion", "Colt"]
    node_num_list = [25,58,125,153]
    bandwidth_setting = [20]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[3] #0,1,2,3

    exp_data = [[],[],[],[]]

    for topology in topology_list:
        file_name_both = 'exp_data/raw_data0704/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file_both = pd.read_excel(file_name_both, sheet_name=sheet_name, usecols=col_name)
        
        exp_list = input_file_both["Main"].tolist()
        exp_data[0].append(sum(exp_list)/len(exp_list))
        exp_list = input_file_both["Merge"].tolist()
        exp_data[1].append(sum(exp_list)/len(exp_list))

        file_name_mergeG = 'exp_data/our_compare/mergeG_only/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file_mergeG = pd.read_excel(file_name_mergeG, sheet_name=sheet_name, usecols=["Merge"])

        exp_list = input_file_mergeG["Merge"].tolist()
        exp_data[2].append(sum(exp_list)/len(exp_list))

        file_name_mergeP = 'exp_data/our_compare/mergeP_only/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file_mergeP = pd.read_excel(file_name_mergeP, sheet_name=sheet_name, usecols=["Merge"])

        exp_list = input_file_mergeP["Merge"].tolist()
        exp_data[3].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/'+sheet_name+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    data = pd.DataFrame({"topology":node_num_list,"Main":exp_data[0],"MergeG":exp_data[2],"MergeP":exp_data[3],"Both":exp_data[1]})
    data.to_csv(file_path, index=False)


def compare_self_dst():
    topology_list = ["Ion"]
    bandwidth_setting = [20]
    dst_ratio = [0.1,0.2,0.4]
    is_group = 1

    sheet_name = title[0]

    exp_data = [[],[],[],[]]

    for r in dst_ratio:
        file_name_both = 'exp_data/raw_data0704/'+topology_list[0]+'_d'+str(r)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file_both = pd.read_excel(file_name_both, sheet_name=sheet_name, usecols=col_name)
        
        exp_list = input_file_both["Main"].tolist()
        exp_data[0].append(sum(exp_list)/len(exp_list))
        exp_list = input_file_both["Merge"].tolist()
        exp_data[1].append(sum(exp_list)/len(exp_list))

        file_name_mergeG = 'exp_data/our_compare/mergeG_only/'+topology_list[0]+'_d'+str(r)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file_mergeG = pd.read_excel(file_name_mergeG, sheet_name=sheet_name, usecols=["Merge"])

        exp_list = input_file_mergeG["Merge"].tolist()
        exp_data[2].append(sum(exp_list)/len(exp_list))

        file_name_mergeP = 'exp_data/our_compare/mergeP_only/'+topology_list[0]+'_d'+str(r)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file_mergeP = pd.read_excel(file_name_mergeP, sheet_name=sheet_name, usecols=["Merge"])

        exp_list = input_file_mergeP["Merge"].tolist()
        exp_data[3].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/'+sheet_name+'_'+topology_list[0]+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    data = pd.DataFrame({"dst_ratio":dst_ratio,"Main":exp_data[0],"MergeG":exp_data[2],"MergeP":exp_data[3],"Both":exp_data[1]})
    data.to_csv(file_path, index=False)

# group_data()
# bw_failed_data()
place_cost_transocer_num_data()
# all_data_network()
# all_data_dst_ratio()
# compare_self_network()
# compare_self_dst()