'''
Consider the position and viedo quality of user to grouping them. 
'''
import random
import math
import sys
import itertools
import matplotlib.pyplot as plt

def cal_distance(dst, center):
    return math.sqrt((dst[1][0] - center[1][0])**2 + (dst[1][1] - center[1][1])**2)

def cal_different(bound, dst, center):
    if (bound[1] - bound[0]) == 0:
        normal_quality = (abs(dst[0] - center[0]) - bound[0])
    else:
        normal_quality = (abs(dst[0] - center[0]) - bound[0]) / (bound[1] - bound[0])
    
    distance = math.sqrt((dst[1][0] - center[1][0])**2 + (dst[1][1] - center[1][1])**2)
    normal_dis = (distance - bound[2]) / (bound[3] - bound[2])

    return math.sqrt(normal_quality**2 + normal_dis**2)

def clustering(dsts, centers):
    cluster = dict()
    for c in range(len(centers)):
        cluster[c] = []

    q_min = [sys.maxsize] * len(centers)
    q_max = [-1] * len(centers)
    dis_min = [sys.maxsize] * len(centers)
    dis_max = [-1] * len(centers)

    for dst in dsts:
        dic = dsts[dst]
        for i,center in enumerate(centers):
            q_min[i] = min(q_min[i],abs(dic[0] - center[0]))
            q_max[i] = max(q_max[i],abs(dic[0] - center[0]))
            
            distance = math.sqrt((dic[1][0] - center[1][0])**2 + (dic[1][1] - center[1][1])**2)
            dis_min[i] = min(dis_min[i],distance)
            dis_max[i] = max(dis_max[i],distance)

    for dst in dsts:
        min_dif = sys.maxsize
        min_center = -1
        dic = dsts[dst]

        for i,center in enumerate(centers):
            bound = (q_min[i], q_max[i], dis_min[i], dis_max[i])
            dif = cal_different(bound, dic, center)
            if dif < min_dif:
                min_dif = dif
                min_center = i
        cluster[min_center].append(dst)
    
    return cluster
        
def get_centers(cluster, client):
    centers = []

    for i in cluster:
        center_quality = 0
        center_x = 0
        center_y = 0
        
        dic = cluster[i]
        for dst in dic:
            center_quality += client[dst][0]
            center_x += client[dst][1][0]
            center_y += client[dst][1][1]
        center = (math.ceil(center_quality/len(dic)),[round(center_x/len(dic),8),round(center_y/len(dic),8)])    
        centers.append(center)
    
    return centers

def print_graph(client, centers, cluster, n):
    color_cycle = itertools.cycle(["#ff7f0e","#2ca02c","#1f77b4"])

    plt.figure()
    for i in cluster:
        x_list = list(client[j][1][0] for j in cluster[i])
        y_list = list(client[j][1][1] for j in cluster[i])
        plt.scatter(x_list, y_list, color=next(color_cycle))
        for j in cluster[i]:
            q = client[j][0]
            plt.annotate(q, (client[j][1][0], client[j][1][1]))

    plt.savefig('g_img/tmp_'+n+'.png')
    plt.show()


def k_means(pos, dsts, video_type, user_limit):
    dst_num = len(dsts)
    k = 3 #math.ceil(math.sqrt(dst_num))

    x_bound_low = min(pos[d].tolist()[0] for d in pos)
    x_bound_high = max(pos[d].tolist()[0] for d in pos)
    y_bound_low = min(pos[d].tolist()[1] for d in pos)
    y_bound_high = max(pos[d].tolist()[1] for d in pos)

    client = dict()
    for dst in dsts:
        q = dsts[dst]
        pos_convert = [round(pos[dst].tolist()[0],8), round(pos[dst].tolist()[1],8)]
        client[dst] = (video_type.index(q), pos_convert)
    
    isFinish = 0
    while (isFinish < k):
        # random samples
        centers = []
        for i in range(k):
            sample = (random.randint(0, len(video_type)), [round(random.uniform(x_bound_low,x_bound_high),8), round(random.uniform(y_bound_low,y_bound_high),8)])
            centers.append(sample)

        # cluster users
        isFinish = 0
        cluster = clustering(client, centers)
        
        for i in cluster:
            if len(cluster[i]) >= user_limit:
                isFinish += 1

    # Calculate new cluster centers
    new_centers = get_centers(cluster, client)

    isConvergence = 0
    max_dif = -1
    count = 0

    # Check if the clustering convergence or not
    while isConvergence == 0:
        max_dif = -1
        for i in range(k):
            distance = math.sqrt((centers[i][1][0] - new_centers[i][1][0])**2 + (centers[i][1][1] - new_centers[i][1][1])**2)
            dif = math.sqrt((centers[i][0] - new_centers[i][0])**2 + distance**2)
            max_dif = max(max_dif, dif)
        
        #print(max_dif)
        if max_dif < 10**-10:
            isConvergence = 1
            break

        centers = new_centers

        cluster = clustering(client, centers)
        
        new_centers = get_centers(cluster, client)

        count += 1
     
    return cluster

