import pandas as pd
import matplotlib.pyplot as plt

title = {0: "Total_cost", 1: "Trans_cost", 2: "Proc_cost cost", 3: "Plac_cost", \
    4: "Avg_cost", 5: "Transcoder_num", 6: "Delay", 7: "Running_time", 8: "failed_num"}
        
# topology_list = ["Agis", "Dfn", "Ion", "Colt"]
node_num = {"Agis":25, "Dfn":58, "Ion":125, "Colt":153}
node_num_list = [25,58,125,153]
# bandwidth_setting = [2, 5, 10, 20]
# exp_factor = [0.1, 0.2, 0.4]
exp_num = 50
# is_group = 0

# sheet_name = "Total_cost"
col_name = ["Unicast","JPR","JPR_notReuse","FirstFit","Main","Merge"]

# bw = 20, dst = 0.4 in every topology with g0, g1
def group_data():
    topology_list = ["Agis", "Dfn", "Ion"]#, "Colt"]
    node_num_list = [25,58,125]
    dst_ratio = 0.4
    bandwidth_setting = [20]
    is_group = 1 #0,1

    sheet_name = title[0]

    exp_data = [[],[],[],[],[],[]]

    for topology in topology_list:
        file_name = 'exp_data/raw_data/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        failed_dst_file = pd.read_excel(file_name, sheet_name="failed_num", usecols=col_name)
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            failed_dst_list = failed_dst_file[col].tolist()
            service_dst = len(exp_list) * node_num[topology] * dst_ratio - sum(failed_dst_list)
            # exp_data[i].append(sum(exp_list)/service_dst)
            exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/group/d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    data = pd.DataFrame({"topology":node_num_list,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main":exp_data[4],"Merge":exp_data[5]})
    data.to_csv(file_path, index=False)

# dst = 0.4, group = 1 in Dfn,Ion with different bandwidth
def bw_failed_data():
    topology_list = ["Dfn", "Ion"]
    bandwidth_setting = [2, 5, 10, 20]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[8]

    for topology in topology_list:
        exp_data = [[],[],[],[],[],[]]
        for bw in bandwidth_setting:
            for i,col in enumerate(col_name):
                file_name = 'exp_data/raw_data/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bw)+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
                input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
            
                exp_list = input_file[col].tolist()
                exp_data[i].append(sum(exp_list)/(len(exp_list)*node_num[topology]*dst_ratio))

        file_path = 'exp_data/bw_fail/'+topology+'_d'+str(dst_ratio)+'_g'+str(is_group)+'.csv'
        data = pd.DataFrame({"bandwidth":bandwidth_setting,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main":exp_data[4],"Merge":exp_data[5]})
        data.to_csv(file_path, index=False)


# bw = 20, dst = 0.4, group = 1 in every topology
def place_cost_transocer_num_data():
    topology_list = ["Agis", "Dfn", "Ion"]
    node_num_list = [25,58,125]
    bandwidth_setting = [20]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[3] #3,5

    exp_data = [[],[],[],[],[],[]]

    for topology in topology_list:
        file_name = 'exp_data/raw_data/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/transcoder_place_cost/'+sheet_name+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    data = pd.DataFrame({"topology":node_num_list,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main":exp_data[4],"Merge":exp_data[5]})
    data.to_csv(file_path, index=False)

# bw = 20, dst = 0.4, group = 1 in every topology
def all_data_network():
    topology_list = ["Agis", "Dfn", "Ion"]
    node_num_list = [25,58,125]
    bandwidth_setting = [10]
    dst_ratio = 0.4
    is_group = 1

    sheet_name = title[0] #0,6,7,8

    exp_data = [[],[],[],[],[],[]]

    for topology in topology_list:
        file_name = 'exp_data/raw_data/'+topology+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        failed_dst_file = pd.read_excel(file_name, sheet_name="failed_num", usecols=col_name)
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            failed_dst_list = failed_dst_file[col].tolist()
            service_dst = len(exp_list) * node_num[topology] * dst_ratio - sum(failed_dst_list)
            exp_data[i].append(sum(exp_list)/service_dst)
            # exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/network/'+sheet_name+'_d'+str(dst_ratio)+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    data = pd.DataFrame({"topology":node_num_list,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main":exp_data[4],"Merge":exp_data[5]})
    data.to_csv(file_path, index=False)

def all_data_dst_ratio():
    topology_list = ["Ion"]
    bandwidth_setting = [10]
    dst_ratio = [0.1,0.2,0.4]
    is_group = 1

    sheet_name = title[0] #0,6,7

    exp_data = [[],[],[],[],[],[]]
    for r in dst_ratio:
        file_name = 'exp_data/raw_data/'+topology_list[0]+'_d'+str(r)+'_bw'+str(bandwidth_setting[0])+'_exp'+str(exp_num)+'_g'+str(is_group)+'.xlsx'
        input_file = pd.read_excel(file_name, sheet_name=sheet_name, usecols=col_name)
        failed_dst_file = pd.read_excel(file_name, sheet_name="failed_num", usecols=col_name)
        
        for i,col in enumerate(col_name):
            exp_list = input_file[col].tolist()
            failed_dst_list = failed_dst_file[col].tolist()
            service_dst = len(exp_list) * node_num[topology_list[0]] * r - sum(failed_dst_list)
            exp_data[i].append(sum(exp_list)/service_dst)
            # exp_data[i].append(sum(exp_list)/len(exp_list))

    file_path = 'exp_data/dst_ratio/'+sheet_name+topology_list[0]+'_bw'+str(bandwidth_setting[0])+'_g'+str(is_group)+'.csv'
    data = pd.DataFrame({"dst_ratio":dst_ratio,"Unicast":exp_data[0],"JPR":exp_data[1],"JPR_notReuse":exp_data[2],"FirstFit":exp_data[3],"Main":exp_data[4],"Merge":exp_data[5]})
    data.to_csv(file_path, index=False)

group_data()
# bw_failed_data()
# place_cost_transocer_num_data()
# all_data_network()
# all_data_dst_ratio()
