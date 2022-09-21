import torch
import numpy as np
import matplotlib.pyplot as plt


def calc_precisions_topn(qB, rB, query_L, retrieval_L, recall_gas=0.02, num_retrieval=5000):
    qB = torch.FloatTensor(qB)
    rB = torch.FloatTensor(rB)
    query_L = torch.FloatTensor(query_L)
    retrieval_L = torch.FloatTensor(retrieval_L)
    num_query = query_L.shape[0]
    # num_retrieval = retrieval_L.shape[0]
    precisions = [0] * int(1 / recall_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(recall_gas, 1 + recall_gas, recall_gas)):
            total = int(num_retrieval * recall)
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            # right_num = torch.nonzero(gnd[: total]).squeeze().shape[0]
            right_num = right.size
            precisions[i] += (right_num/total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions



def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def PR_curve(precisions: np.ndarray, label: list, title: str, x=None):
    # bit = precisions.shape[1]

    # min_presion = np.min([np.min(l) for l in precisions])
    # max_presion = np.max([np.max(l) for l in precisions])
    min_presion = 0.5
    max_presion = 1
    plt.title(title)
    plt.xticks(np.arange(0.1, 1.1, 0.1))
    plt.xlabel("recall")
    plt.yticks(np.arange(round(min_presion * 10 - 1) * 0.1, (round(max_presion * 10)+1) * 0.1, 0.1))
    plt.ylabel("precision")
    if x is None:
        x = np.arange(0.02, 1.02, 0.02)
        # x = np.expand_dims(x, precisions.shape)
    colors = ['red', 'blue', 'c', 'green', 'yellow', 'black', 'lime', 'grey', 'pink', 'navy']
    markets = ['o', 'v', '^', '>', '<', '+', 'x', '*', 'd', 'D']
    for i in range(precisions.shape[0]):
        # plt.plot(x[i], precisions[i, :], marker=markets[i % 10], color=colors[i % 10], label=label[i])
        plt.plot(x[i], precisions[i], color=colors[i % 10], label=label[i])
        # plt.plot(x, precisions[i, :], color=colors[i % 10], label=label[i])
    plt.grid()
    ax = plt.axes()
    ax.set(xlim=(0, 1), ylim=(round(min_presion * 10 - 1) * 0.1, (round(max_presion * 10)) * 0.1))
    plt.legend()
    # plt.axes('tight')
    plt.show()


def calc_value(qu_BI, qu_BT, qu_L, re_BI, re_BT, re_L):
    i2ts = t2is = []
    i2t = calc_precisions_topn(qu_BI, re_BT, qu_L, re_L, 5000)
    t2i = calc_precisions_topn(qu_BT, re_BI, qu_L, re_L, 5000)
    i2ts.append(i2t)
    t2is.append(t2i)
    return i2t, t2i

def draw_PR(qu_BI, qu_BT, qu_L, re_BI, re_BT, re_L):
    labels = ["FARH"]
    i2t, t2i = calc_value(qu_BI, qu_BT, qu_L, re_BI, re_BT, re_L)
    PR_curve(i2t, labels)
    PR_curve(t2i, labels)