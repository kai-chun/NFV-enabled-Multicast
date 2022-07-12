import benchmark_main as main
import benchmark_main1 as main1

topology_list = ["Agis","Dfn","Ion","Colt"]
bandwidth_setting = [20]#[2, 5, 10, 20]
exp_num = 50
is_group = 1 # If is_group = 0, execute k_means with k = 1.
gen_new_data = 0 # If gen_new_data = 1, generate new test data

for bandwidth in bandwidth_setting:
    for topology in topology_list:
        main1.main(topology, bandwidth, exp_num, is_group, gen_new_data)