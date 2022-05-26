import benchmark_main as main

topology_list = ["Agis", "Dfn"]#, "Ion", "Colt"]
bandwidth_setting = [2, 5, 10, 20]
exp_num = 50
is_group = 0 # If is_group = 0, execute k_means with k = 1.
gen_new_data = 1 # If gen_new_data = 1, generate new test data

for bandwidth in bandwidth_setting:
    for topology in topology_list:
        main.main(topology, bandwidth, exp_num, is_group, gen_new_data)