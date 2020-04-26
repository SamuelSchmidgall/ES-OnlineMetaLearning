import gym
import numpy as np
from copy import deepcopy
import MNIST.mnist as mnist
from Networks.utils import *
from multiprocessing import Pool
from Networks.network_modules_numpy import NetworkModule
from online_continual.networks_numpy_continual import *


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  # only difference


class ContinualMNISTEnv:
    def __init__(self):
        self.labels = 3
        self.label_itr = 0
        self.presentation_t = 3
        self.presentation_itr = 0

        # training_images, training_labels, test_images, test_labels
        self.mnist = mnist.load()
        self.label_ordering = np.array(
            [_ for _ in range(self.labels)])
        np.random.shuffle(self.label_ordering)
        self.label_samples = {str(_):[_k for _k in range(len(self.mnist[1]))
            if self.mnist[1][_k] == _] for _ in range(self.labels)}

        self.action_space = np.zeros((self.labels + 1, 1))
        self.observation_space = np.zeros((28*28 + 1, 1))

    def reset(self):
        self.label_itr = 0
        self.presentation_itr = 0
        np.random.shuffle(self.label_ordering)
        random_sample = self.mnist[0][np.random.choice(
            self.label_samples[str(self.label_itr)])].flatten()/255.0
        random_sample = np.append(np.array(self.label_itr), random_sample)
        return random_sample

    def step(self):
        random_sample = self.mnist[0][np.random.choice(
            self.label_samples[str(self.label_itr)])].flatten()/255.0
        random_sample = np.append(np.array(self.label_itr), random_sample)
        self.presentation_itr = (self.presentation_itr + 1)%self.presentation_t
        if self.presentation_itr == 0:
            self.label_itr = (self.label_itr + 1)%self.labels
        if self.presentation_itr == 0 and self.label_itr == 0:
            return random_sample, True
        return random_sample, False


import pickle

with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/online_continual/plast_model.pkl", 'rb') as f:
    network = pickle.load(f)

local_env = ContinualMNISTEnv()
state = local_env.reset()

acc = list()

for _k in range(100):
    accuracy = 0.0
    # forward propagate using noisy weights and plasticity values, also update trace
    state = state.reshape((1, state.size))
    label = local_env.label_ordering[local_env.label_itr]
    auto_enc = network.forward(state)[0][local_env.labels]
    # interact with environment
    # state, reward, game_over, _ = local_env.step(action)
    state, game_over = local_env.step()

    # end sim iteration if termination state reached
    if game_over:
        # evaluation time
        for _eval in local_env.label_ordering:
            img = local_env.mnist[0][np.random.choice(
                local_env.label_samples[str(_eval)])].flatten() / 255.0
            img = np.append(np.array(-1), img)
            v = softmax(network.forward(img)[0][:local_env.labels])
            action = np.argmax(v)

            if _eval == action:
                accuracy += 1

        acc.append(accuracy/3.0)
        network.reset()
        state = local_env.reset()

print(sum(acc)/len(acc))





