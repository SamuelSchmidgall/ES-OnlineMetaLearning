import os
import gym
import time
import logging
import numpy as np
from gym import wrappers
from copy import deepcopy
from multiprocessing import Pool
from collections import namedtuple


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Adam:
    def __init__(self, params, stepsize, epsilon=1e-08, beta1=0.99, beta2=0.999):
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.params = params
        self.dim = params.size
        self.epsilon = epsilon
        self.stepsize = stepsize
        self.m = np.zeros(params.size, dtype=np.float32)
        self.v = np.zeros(params.size, dtype=np.float32)

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.params
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        new_theta = self.params + step
        self.params = new_theta
        return ratio, new_theta

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step



class NaturalEvolutionaryStrategy:
    def __init__(self, learning_rate, noise_std, num_eps_samples, environment, target_env_interacts, inp_dim,
                 outp_dim, lr_decay=0.9999, lr_limit=0.001, std_decay=0.999, std_limit=0.01, discrete=False,
                 env_name=None, act_noise_std=None, layer=None, rollout_avg=5, plasticity=None):
        self.inp_d = inp_dim
        self.outp_d = outp_dim
        self.env_name = env_name
        self.lr_limit = lr_limit
        self.lr_decay = lr_decay
        self.noise_std = noise_std
        self.std_decay = std_decay
        self.std_limit = std_limit
        self.plasticity = plasticity
        self.total_env_interacts = 0
        self.max_env_interacts = 1000
        self.environment = environment
        self.discrete_action = discrete
        self.learning_rate = learning_rate
        self.num_env_rollouts = rollout_avg
        self.action_noise_std = act_noise_std
        self.num_eps_samples = num_eps_samples
        self.target_env_interacts = target_env_interacts

        antithetic = True
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.num_eps_samples % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.num_eps_samples / 2)

        params = list()
        self.hebb = True
        self.stepsize = 1
        if self.plasticity is None:
            self.plasticity = [False for _ in range(len(layer)+1)]

        self.biases = list()
        layer = [self.inp_d] + layer + [self.outp_d]
        for _i in range(len(layer)-1):
            self.biases.append(np.zeros((1, layer[_i+1])))
            params.append(self.biases[-1].flatten())

        self.weights = list()
        for _i in range(len(layer)-1):
            self.weights.append(np.zeros((layer[_i], layer[_i+1])))
            params.append(self.weights[-1].flatten())

        self.hebb_template = list()
        self.fan_in_biases = list()
        self.fan_in_weights = list()
        self.fan_out_biases = list()
        self.fan_out_weights = list()
        self.alpha_plasticity = list()

        for _i in range(len(plasticity)):
            if self.plasticity[_i]:
                self.alpha_plasticity.append(np.zeros((layer[_i], layer[_i+1])))
                self.fan_in_biases.append(np.zeros((1, 1)))
                self.fan_in_weights.append(np.zeros((layer[_i+1], 1)))
                self.fan_out_biases.append(np.zeros((1, layer[_i+1])))
                self.fan_out_weights.append(np.zeros((1, layer[_i+1])))
                self.hebb_template.append(np.zeros((layer[_i], layer[_i+1])))

                params.append(self.alpha_plasticity[-1].flatten())
                params.append(self.fan_in_biases[-1].flatten())
                params.append(self.fan_in_weights[-1].flatten())
                params.append(self.fan_out_biases[-1].flatten())
                params.append(self.fan_out_weights[-1].flatten())

        self.params = np.concatenate(params)
        self.optimizer = Adam(self.params, learning_rate)

    def update_params(self, new_params):
        eps_index = 0
        self.params[:] = new_params[:]

        # compute bias updates
        for _b in range(len(self.biases)):
            self.biases[_b] = new_params[eps_index:eps_index + self.biases[_b].size].reshape(self.biases[_b].shape)
            eps_index += self.biases[_b].size

        # compute weight updates
        for _w in range(len(self.weights)):
            self.weights[_w] = new_params[eps_index:eps_index + self.weights[_w].size].reshape(self.weights[_w].shape)
            eps_index += self.weights[_w].size

        param_itr = 0
        for _w in range(len(self.weights)):
            if self.plasticity[_w]:
                self.alpha_plasticity[param_itr] = new_params[eps_index:eps_index
                    + self.alpha_plasticity[param_itr].size].reshape(self.alpha_plasticity[param_itr].shape)
                eps_index += self.alpha_plasticity[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.weights)):
            if self.plasticity[_w]:
                self.fan_in_biases[param_itr] = new_params[eps_index:eps_index
                    + self.fan_in_biases[param_itr].size].reshape(self.fan_in_biases[param_itr].shape)
                eps_index += self.fan_in_biases[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.weights)):
            if self.plasticity[_w]:
                self.fan_in_weights[param_itr] = new_params[eps_index:eps_index
                    + self.fan_in_weights[param_itr].size].reshape(self.fan_in_weights[param_itr].shape)
                eps_index += self.fan_in_weights[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.weights)):
            if self.plasticity[_w]:
                self.fan_out_biases[param_itr] = new_params[eps_index:eps_index
                    + self.fan_out_biases[param_itr].size].reshape(self.fan_out_biases[param_itr].shape)
                eps_index += self.fan_out_biases[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.weights)):
            if self.plasticity[_w]:
                self.fan_out_weights[param_itr] = new_params[eps_index:eps_index
                    + self.fan_out_weights[param_itr].size].reshape(self.fan_out_weights[param_itr].shape)
                eps_index += self.fan_out_weights[param_itr].size
                param_itr += 1

    def anneal_lr(self):
        self.learning_rate = max(self.lr_limit, self.learning_rate * self.lr_decay)

    def anneal_std(self):
        self.noise_std = max(self.std_limit, self.noise_std * self.std_decay)

    def hebb_sample(self, seed=None):
        if type(seed) is tuple:
            seed = seed[1]
        if seed is not None:
            rand_m = np.random.RandomState(seed)
        else:
            rand_m = np.random.RandomState()

        # antithetic sampling
        if self.antithetic:
            epsilon_half = rand_m.randn(self.half_popsize, self.params.size)
            sample = np.concatenate([epsilon_half, - epsilon_half])*self.noise_std
        else:
            sample = rand_m.randn(self.num_eps_samples, self.params.size)*self.noise_std
        return sample

    def hebb_forward(self, x, weight, bias, h_comp, hebb):
        """
        Compute forward propagation through plastic network
        :param x: (ndarray) state input variable
        :param weight: (ndarray) feedforward network weight values
        :param bias: (ndarray) feedforward network weight biases
        :param h_comp: (tuple(ndarray[x5])) list of relevant hebbian parameters
        :param hebb: (ndarray) recurrent hebbian trace
        :return: (tuple(ndarray, ndarray)) action and post-processed hebbian trace
        """
        # reshape input for propagation
        x = np.reshape(x, newshape=(1, x.size))

        # decompress hebbian components
        alpha_hebb, fan_in_bias,\
          fan_in_weight, fan_out_bias, fan_out_weight = h_comp

        # compute initial linear feedforward
        for _weight in range(len(weight) - 1):
            pre_syn = deepcopy(x)
            if self.plasticity[_weight]:
                h = np.matmul(pre_syn, np.multiply(alpha_hebb[_weight], hebb[_weight]))
                x = np.tanh(np.matmul(x, weight[_weight]) + bias[_weight] + h)
                post_syn = deepcopy(x)
                # fan in eta
                eta = np.tanh(np.matmul(post_syn, fan_in_weight[_weight]) + fan_in_bias[_weight])
                # fan out eta
                eta = np.tanh(np.matmul(eta, fan_out_weight[_weight]) + fan_out_bias[_weight])
                # update hebbian trace
                hebb[_weight] = np.matmul(pre_syn.transpose(), eta * post_syn) + hebb[_weight]
                hebb[_weight] = np.clip(hebb[_weight], a_min=-1, a_max=1)
            else:
                x = np.tanh(np.matmul(x, weight[_weight]) + bias[_weight])

        pre_syn = deepcopy(x)
        if self.plasticity[-1]:
            h = np.matmul(pre_syn, np.multiply(alpha_hebb[-1], hebb[-1]))
            x = np.tanh(np.matmul(x, weight[-1]) + bias[-1] + h)
            post_syn = deepcopy(x)
            # fan in eta
            eta = np.tanh(np.matmul(post_syn, fan_in_weight[-1]) + fan_in_bias[-1])
            # fan out eta
            eta = np.tanh(np.matmul(eta, fan_out_weight[-1]) + fan_out_bias[-1])
            # update hebbian trace
            hebb[-1] = np.matmul(pre_syn.transpose(), eta * post_syn) + hebb[-1]
            hebb[-1] = np.clip(hebb[-1], a_min=-1, a_max=1)
        else:
            x = np.matmul(x, weight[-1]) + bias[-1]

        if self.action_noise_std is not None:
            x += np.random.randn(*x.shape)*self.action_noise_std

        return x, hebb

    def generate_weights(self, noise):
        eps_index = 0
        biases = list()
        weights = list()
        alpha_plasticity = list()
        fan_in_biases = list()
        fan_in_weights = list()
        fan_out_biases = list()
        fan_out_weights = list()

        # compute bias updates
        for _b in range(len(self.biases)):
            bias = self.biases[_b] + noise[eps_index:eps_index
                + self.biases[_b].size].reshape(self.biases[_b].shape)
            eps_index += self.biases[_b].size
            biases.append(bias)

        for _w in range(len(self.weights)):
            weight = self.weights[_w] + noise[eps_index:eps_index
                + self.weights[_w].size].reshape(self.weights[_w].shape)
            weights.append(weight)
            eps_index += self.weights[_w].size

        param_itr = 0
        for _w in range(len(self.biases)):
            if self.plasticity[_w]:
                a_p = self.alpha_plasticity[param_itr] + noise[eps_index:eps_index
                    + self.alpha_plasticity[param_itr].size].reshape(self.alpha_plasticity[param_itr].shape)
                alpha_plasticity.append(a_p)
                eps_index += self.alpha_plasticity[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.biases)):
            if self.plasticity[_w]:
                fi_b = self.fan_in_biases[param_itr] + noise[eps_index:eps_index
                    + self.fan_in_biases[param_itr].size].reshape(self.fan_in_biases[param_itr].shape)
                fan_in_biases.append(fi_b)
                eps_index += self.fan_in_biases[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.biases)):
            if self.plasticity[_w]:
                fi_w = self.fan_in_weights[param_itr] + noise[eps_index:eps_index
                    + self.fan_in_weights[param_itr].size].reshape(self.fan_in_weights[param_itr].shape)
                fan_in_weights.append(fi_w)
                eps_index += self.fan_in_weights[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.biases)):
            if self.plasticity[_w]:
                fo_b = self.fan_out_biases[param_itr] + noise[eps_index:eps_index
                    + self.fan_out_biases[param_itr].size].reshape(self.fan_out_biases[param_itr].shape)
                fan_out_biases.append(fo_b)
                eps_index += self.fan_out_biases[param_itr].size
                param_itr += 1

        param_itr = 0
        for _w in range(len(self.biases)):
            if self.plasticity[_w]:
                fo_w = self.fan_out_weights[param_itr] + noise[eps_index:eps_index
                    + self.fan_out_weights[param_itr].size].reshape(self.fan_out_weights[param_itr].shape)
                fan_out_weights.append(fo_w)
                eps_index += self.fan_out_weights[param_itr].size
                param_itr += 1

        return weights, biases, (alpha_plasticity, fan_in_biases, fan_in_weights, fan_out_biases, fan_out_weights)

    def compute_weight_decay(self, weight_decay, model_param_list):
        model_param_grid = np.array(model_param_list + self.params)
        return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

    def compute_hebb_returns(self, eps_samples, seed=None, environment=None):
        """
        (Hebb specific) Compute return values for respective epsilon samples
        :param eps_samples: (ndarray) epsilon noise tensors
        :param seed: (int) seed value used to initialize random seed
        :param environment: (gym.env) gym environment to interact with
        :return: (tuple(ndarray, float)) list of returns and average time until termination
        """
        local_env = environment  # gym.make(self.env_name)

        # initialize random generator
        if seed is not None:
            rand_m = np.random.RandomState(seed)
        else:
            rand_m = np.random.RandomState()

        avg_stand = 0
        returns = list()
        total_env_interacts = 0
        # iterate through the noise samples
        for _sample in range(self.num_eps_samples):
            return_avg = 0.0
            weights, biases, h_comp =\
                self.generate_weights(eps_samples[_sample])
            # iterate through the target number of env rollouts
            for _roll in range(self.num_env_rollouts):
                state = local_env.reset()
                trace = deepcopy(self.hebb_template)
                # run the simulation until it terminates or max env interation terminates it
                for _inter in range(self.max_env_interacts):
                    # forward propagate using noisy weights and plasticity values, also update trace
                    action, trace = self.hebb_forward(state, weight=weights,
                        bias=biases, h_comp=h_comp, hebb=trace)
                    action = action[0]
                    action = np.clip(action, a_min=-1, a_max=1)
                    # if discrete action space then transform
                    if self.discrete_action:
                        action_probabilities = softmax(action)
                        action = rand_m.choice(
                            np.array(list(range(len(action_probabilities)))), p=action_probabilities)
                    # interact with environment
                    state, reward, game_over, _ = local_env.step(action)
                    return_avg += reward
                    # end sim iteration if termination state reached
                    if game_over:
                        break
                    avg_stand += 1
                    total_env_interacts += 1
            returns.append(return_avg/self.num_env_rollouts)
        return np.array(returns), total_env_interacts


class ParallelNaturalEvolutionaryStrategy:
    def __init__(self, natural_evo_strategy, environment_title, novelty=None, workers=8, pop_size=128):
        self.novelty = novelty
        self.workers = workers
        self.nes = natural_evo_strategy
        self.population_total = pop_size
        self.population_per_worker = pop_size // self.workers
        self.environments = [gym.make(environment_title) for _ in range(workers)]

    def parallel_returns(self, x):
        worker_id, seed = x
        return self.nes.compute_hebb_returns(
            self.nes.hebb_sample(seed=seed), seed=seed, environment=self.environments[worker_id])

    def update(self):
        seeds = [(_, np.random.randint(1, 999999)) for _ in range(self.workers)]
        with Pool(self.workers) as p:
            values = p.map(func=self.parallel_returns, iterable=seeds)

        samples = list()
        total_timesteps = 0
        timestep_list = list()
        sample_returns = list()

        for _worker in range(self.workers):
            seed = seeds[_worker]
            returns, timesteps = values[_worker]

            timestep_list.append(timesteps)
            total_timesteps += timesteps
            sample_returns += returns.tolist()
            samples += [self.nes.hebb_sample(seed)]

        eps = np.concatenate(samples)
        returns = np.array(sample_returns)
        self.nes.num_eps_samples = self.population_total

        # rank sort and convert rewards
        ret_len = len(returns)
        returns = [[_r, returns[_r], 0] for _r in range(ret_len)]
        returns.sort(key=lambda x: x[1], reverse=True)
        for _r in range(ret_len):
            returns[_r][2] = ((-_r + ret_len // 2) / (ret_len // 2)) / 2
        returns.sort(key=lambda x: x[0])
        returns = np.array([_r[2] for _r in returns])

        self.nes.weight_decay = 0.01
        if self.nes.weight_decay > 0:
            l2_decay = self.nes.compute_weight_decay(self.nes.weight_decay, eps)
            returns += l2_decay

        returns = (returns - np.mean(returns)) / (returns.std() + 1e-5)

        # compute weight update
        # sigma squared to account for sample = eps*sigma to get rid of sigma
        change_mu = (1/(self.population_total*(self.nes.noise_std**2)))*np.dot(eps.T, returns)

        ratio, theta = self.nes.optimizer.update(-change_mu)
        self.nes.update_params(theta)

        self.nes.num_eps_samples = self.population_per_worker
        avg_return_rec = sum(sample_returns) / len(sample_returns)

        # anneal if necessary
        self.nes.anneal_lr()
        self.nes.anneal_std()

        return total_timesteps, avg_return_rec, \
               sum(timestep_list)/(self.nes.num_env_rollouts*self.population_total*self.nes.num_env_rollouts)


"""
To understand whether we can
scale to more complex locomotion tasks with higher dimensionality, the last family of tasks involves a simulated
quadruped (“ant”) tasked to walk to randomly placed goals
(see Fig. 5), in a similar setup as the wheeled robot. This task
presents a further exploration challenge, since only carefully
coordinated leg motion actually produces movement to different positions in the world, so an ideal exploration strategy
would always walk, but would walk to different places. At
meta-training time, the agent receives the negative distance
to the goal as the reward, while at meta-test time, reward is
only given within a small distance of the goal.
"""

import pickle

if __name__ == '__main__':

    env_title = "Pendulum-v0"
    env = gym.make(env_title)

    discrete_action = type(env.action_space) == gym.spaces.Discrete

    try:
        outp_sz = env.action_space.shape[0]
    except Exception:
        outp_sz = env.action_space.n

    inp_sz = env.observation_space.shape[0]

    novelty_search= {
        "k": 10,
        "use_ns": True,
        "population_size": 5,
        "num_rollouts": 5,
        "selection_method": "novelty_prob"
    }

    roll_avg = 2
    env_int = 1000
    num_workers = 2
    target_pop = 48
    layers = [64]
    plastic = [False, False]
    pop = (target_pop // num_workers)*num_workers
    population = pop // num_workers

    noise_standard = 0.01
    noise_limit = 0.01
    noise_decay = 1.0

    init_learning_rate = 0.01
    min_learning_rate = 0.01
    learn_decay = 1.0

    action_noise_std = 0.0

    train_times = 10




    for _t in range(train_times):
        rewards = list()
        t_timesteps = 0
        es = NaturalEvolutionaryStrategy(learning_rate=init_learning_rate,
            std_limit=noise_limit, std_decay=noise_decay, noise_std=noise_standard,
            num_eps_samples=population, environment=env, target_env_interacts=env_int,
            lr_decay=learn_decay, discrete=discrete_action, inp_dim=inp_sz, outp_dim=outp_sz,
            env_name=env_title, act_noise_std=action_noise_std, layer=layers,
            rollout_avg=roll_avg, plasticity=plastic)

        parallel_nes = ParallelNaturalEvolutionaryStrategy(
            natural_evo_strategy=es, workers=num_workers,
            pop_size=pop, environment_title=env_title, novelty=novelty_search)
        for _itr in range(env_int):
            time_taken, avg_reward_rec, stand_time = parallel_nes.update()
            t_timesteps += time_taken
            rewards.append((_itr, t_timesteps, avg_reward_rec))

            print("Iteration: {}".format(_itr))
            print("Total Timesteps: {}".format(t_timesteps))
            print("Average Stand Time: {}".format(stand_time))
            print("Average Return: {}".format(avg_reward_rec))
            print("~" * 30)

            if _itr % 5 == 0:
                with open("DBhb_nes1{}.pkl".format(_t), "wb") as f:
                    pickle.dump(rewards, f)

                #with open("hebb_nes_weights.pkl", "wb") as f:
                #    pickle.dump(parallel_nes.nes.return_weights(), f)