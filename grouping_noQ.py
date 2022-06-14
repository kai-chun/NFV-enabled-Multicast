'''
Consider the position and viedo quality of user to grouping them. 
'''
import random
import math
import sys
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Calculate Standard Deviation of distance between dst to all other dsts
def cal_distance_SD(G, dst, client):
    distance_list = []
    for c in client:
        distance = nx.shortest_path_length(G, source=dst, target=c)
        distance_list.append(distance)
    return np.std(distance_list)

# Calculate sum of distance between dst to all other dsts
def cal_distance_sum(G, dst, client):
    distance = 0
    for c in client:
        distance += nx.shortest_path_length(G, source=dst, target=c)
    return distance

# Calculate difference between dst and center
def cal_different(G, bound, dst, center):
    distance = nx.shortest_path_length(G, source=dst, target=center)
    if (bound[1] - bound[0]) == 0:
        normal_dis = abs(distance - bound[0])
    else:
        normal_dis = abs(distance - bound[0]) / (bound[1] - bound[0])

    return math.sqrt(normal_dis**2)

# Cluster dsts with its minimum difference center
def clustering(G, dsts, centers):
    cluster = dict()
    for c in range(len(centers)):
        cluster[c] = []

    dis_min = [sys.maxsize] * len(centers)
    dis_max = [-1] * len(centers)

    for dst in dsts:
        dic = dsts[dst]
        for i,center in enumerate(centers):
            distance = nx.shortest_path_length(G, source=dst, target=center)
            dis_min[i] = min(dis_min[i],distance)
            dis_max[i] = max(dis_max[i],distance)

    for dst in dsts:
        min_dif = sys.maxsize
        min_center = -1
        dic = dsts[dst]

        for i,center in enumerate(centers):
            bound = (dis_min[i], dis_max[i])
            dif = cal_different(G, bound, dic, center)
            if dif < min_dif:
                min_dif = dif
                min_center = i
        cluster[min_center].append(dst)
    
    return cluster
        
# Get new center in cluster
# center has the average quality and the minimum standard deviation distance to every dst
def get_centers(G, cluster, client):
    centers = []

    for i in cluster:
        center_dis = {}
        
        dic = cluster[i]

        if len(dic) == 0:
            return []

        for dst in G.nodes():
            center_dis[dst] = cal_distance_sum(G, dst, dic)
        
        # Select the minimum standard deviation dst to become new center
        center = min(center_dis, key=center_dis.get)  
        centers.append(center)
    
    return centers

def print_graph(G, cluster):
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    plt.figure()
    pos = nx.spring_layout(G)
    options = {"node_size": 30, "linewidths": 0}
    nx.draw(G, pos, node_color='black',edge_color='gray',**options)

    for i in range(len(cluster)):
        nx.draw_networkx_nodes(G, pos, nodelist=cluster[i], node_color=color_list[i],**options)

    plt.savefig('exp_data/tmp.png')
    plt.close()

# Main grouping algorithm with 2D K-means algorithm
# consider the video quality and shortest path length
def k_means(G, dsts, video_type, user_limit, is_group):
    dst_num = len(dsts)
    if dst_num <= 5:
        k = 1    
    else:
        # k = math.ceil(math.sqrt(dst_num))
        k = math.floor(dst_num/5)

    if is_group == 0:
        k = 1

    G_node = G.nodes()

    client = dict()
    for dst in dsts:
        client[dst] = dst
    
    isFinish = 0
    while isFinish == 0:
        # Random samples
        centers = []
        for i in range(k):
            sample = random.randint(0, len(G_node)-1)
            centers.append(sample)

        # Cluster users
        cluster = clustering(G, client, centers)

        if min(len(cluster[i]) for i in cluster) <= user_limit:
            continue
        
        # Calculate new cluster centers
        new_centers = get_centers(G, cluster, client)

        isConvergence = 0

        # Check if the clustering convergence or not
        while isConvergence == 0:
            max_dif = -1
            for i in range(k-1):
                distance = nx.shortest_path_length(G, source=centers[i], target=new_centers[i])
                dif = math.sqrt(distance**2)
                max_dif = max(max_dif, dif)
            
            if max_dif < 10**-10:
                isConvergence = 1
                ### print
                # for j in range(len(centers)):
                #     print("center",j,":", centers[j][1])
                #     for i in cluster[j]:
                #         print(i,",q=",client[i][0],",dis=", nx.shortest_path_length(G, source=centers[j][1], target=i))
                # print_graph(G, cluster)
                break

            centers = new_centers

            cluster = clustering(G, client, centers)

            if min(len(cluster[i]) for i in cluster) <= user_limit:
                continue
            
            new_centers = get_centers(G, cluster, client)
        
        if min(len(cluster[i]) for i in cluster) <= user_limit:
            k -= 1
            if k < 1: k = 1
            isFinish = 0
        else:
            isFinish = 1

    return cluster