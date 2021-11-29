import random


'''
Random the set of clients of multicast task.

clients = (dsts, video_quality)
'''
def create_client(num, node, src, quality_list):
    clients = []
    
    tmp_N = node
    tmp_N.remove(src)
    dsts = random.sample(tmp_N, num)
    
    for i in range(num):
        clients.append((dsts[i], random.choice(quality_list)))
    
    return clients
