import gym
import pybullet_envs
from Networks.networks_numpy import *


env_id = "CrippledAnt-v0"
envrn = gym.make(env_id)

import pickle
with open("/home/sam/PycharmProjects/ES-OnlineMetaLearning/Networks/data/tmp_single_legant/save_ESnetWALKgatedant1.pkl", "rb") as f:
    agent = pickle.load(f)

state = envrn.reset()

activity_data = {None if _ ==0 else _-1: list() for _ in range(len(envrn.env.cripple_prob))}
seen_data = [envrn.env.crippled_leg_id]

while True:
    #envrn.render()
    state = state.reshape((1, state.size))
    action = agent.forward(state, act=True)
    action, activity = action
    action = action[0]
    activity_data[envrn.env.crippled_leg_id].append(activity)
    state, _, done, _ = envrn.step(action)
    if done:
        if len(seen_data) > 4:
            print("done")
            break
        state = envrn.reset()
        while envrn.env.crippled_leg_id in seen_data:
            state = envrn.reset()
        seen_data.append(envrn.env.crippled_leg_id)


with open("gated_activity.pkl", "wb") as f:
    agent = pickle.dump(activity_data, f)





















