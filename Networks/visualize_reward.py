

import pickle

with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/gated/sv_data/save_rewardgated0.pkl", "rb") as f:
    data = pickle.load(f)
with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/gated/sv_data/save_rewardgated1.pkl", "rb") as f:
    data1 = pickle.load(f)
with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/gated/sv_data/save_rewardgated2.pkl", "rb") as f:
    data2 = pickle.load(f)

import matplotlib.pyplot as plt

x = [_[1] for _ in data]
y = [_[0] for _ in data]

x1 = [_[1] for _ in data1]
y1 = [_[0] for _ in data1]

x2 = [_[1] for _ in data2]
y2 = [_[0] for _ in data2]

m_len = min(min(len(x), len(x1)), len(x2))
x_avg = [[y[_], y1[_], y2[_]] for _ in range(m_len)]
x_avg = [sum(l)/len(l) for l in x_avg]

plt.plot(list(range(m_len)), x_avg)

with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/nongated/sv_data/save_reward2.pkl", "rb") as f:
    d1 = pickle.load(f)
with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/nongated/sv_data/save_reward21.pkl", "rb") as f:
    d12 = pickle.load(f)
with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/nongated/sv_data/save_reward22.pkl", "rb") as f:
    d13 = pickle.load(f)

xd1 = [_[1] for _ in d1]
yd1 = [_[0] for _ in d1]

xd12 = [_[1] for _ in d12]
yd12 = [_[0] for _ in d12]

xd13 = [_[1] for _ in d13]
yd13 = [_[0] for _ in d13]

# todo: next do one with the dropout

m_len = min(min(len(xd1), len(xd12)), len(xd13))
x_avg = [[yd1[_], yd12[_], yd13[_]] for _ in range(m_len)]
x_avg = [sum(l)/len(l) for l in x_avg]


plt.plot(list(range(m_len)), x_avg)
#plt.plot(x2, y2)

plt.show()








