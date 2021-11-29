'''
calculate the weight of G.link
'''

def cal_link_weight(G, edge, alpha, beta, data_rate):
    weight = alpha * (data_rate / G.edges[edge]['bandwidth'] + 1) + beta * (data_rate / G.nodes[edge[1]]['capacity'])
    return weight


