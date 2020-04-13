import numpy as np
from copy import deepcopy
from Networks.utils import *

from multiprocessing import Pool
from Networks.network_modules_numpy import NetworkModule, Parameter



def compute_returns(seed, environment, network, num_eps_samples, num_env_rollouts1=10, num_env_rollouts2=5):
    avg_stand = 0
    returns = list()
    local_env = environment
    total_env_interacts = 0
    max_env_interacts = 1000
    network_cpy = deepcopy(network)
    eps_samples = network.generate_eps_samples(seed, num_eps_samples)

    # iterate through the noise samples
    for _sample in range(eps_samples.shape[0]):
        return_avg = 0.0
        network = deepcopy(network_cpy)
        network.update_params(eps_samples[_sample])

        for _roll2 in range(num_env_rollouts2):
            network.reset_lifetime()
            # iterate through the target number of env rollouts
            for _roll1 in range(num_env_rollouts1):
                network.reset_run(0.0)
                state = local_env.reset()
                # run the simulation until it terminates or max env interation terminates it
                for _inter in range(max_env_interacts):
                    # forward propagate using noisy weights and plasticity values, also update trace
                    state = state.reshape((1, state.size))
                    action = network.forward(state)[0]
                    action = np.clip(action, a_min=-1, a_max=1)
                    # interact with environment
                    state, reward, game_over, _ = local_env.step(action)
                    return_avg += reward
                    # end sim iteration if termination state reached
                    if game_over:
                        break
                    avg_stand += 1
                    total_env_interacts += 1
        returns.append(return_avg / (num_env_rollouts1*num_env_rollouts2))
    return np.array(returns), total_env_interacts



class ParamSampler:
    def __init__(self, sample_type, num_eps_samples, noise_std=0.01):
        """
        Evolutionary Strategies Optimizer
        :param sample_type: (str) type of noise sampling
        :param num_eps_samples: (int) number of noise samples to generate
        :param noise_std: (float) nosie standard deviation
        """
        self.noise_std = noise_std
        self.sample_type = sample_type
        self.num_eps_samples = num_eps_samples
        if self.sample_type == "antithetic":
            assert (self.num_eps_samples % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.num_eps_samples / 2)

    def sample(self, params, seed, num_eps_samples):
        """
        Sample noise for network parameters
        :param params: (ndarray) network parameters
        :param seed: (int) random seed used to sample
        :param num_eps_samples (int) number of noise samples to evaluate
        :return: (ndarray) sampled noise
        """
        if seed is not None:
            rand_m = np.random.RandomState(seed)
        else:
            rand_m = np.random.RandomState()

        sample = None
        if self.sample_type == "antithetic":
            epsilon_half = rand_m.randn(num_eps_samples//2, params.size)
            sample = np.concatenate([epsilon_half, - epsilon_half]) * self.noise_std
        elif self.sample_type == "normal":
            sample = rand_m.randn(num_eps_samples, params.size) * self.noise_std

        return sample


class ESNetwork:
    def __init__(self, params, noise_std=0.01,
            num_eps_samples=64, sample_type="antithetic"):
        self.params = params
        self.es_optim = ParamSampler(noise_std=noise_std,
            sample_type=sample_type, num_eps_samples=num_eps_samples)

    def parameters(self):
        """
        Return list of network parameters
        :return: (ndarray) network parameters
        """
        params = list()
        for _param in range(len(self.params)):
            params.append(self.params[_param].params())
        return np.concatenate(params, axis=0)

    def generate_eps_samples(self, seed, num_eps_samples):
        """
        Generate noise samples for list of parameters
        :param seed: (int) random number seed
        :param num_eps_samples (int) number of noise samples to evaluate
        :return: (ndarray) parameter noise
        """
        params = self.parameters()
        sample = self.es_optim.sample(params, seed, num_eps_samples)
        return sample

    def update_params(self, eps_sample, add_eps=True):
        """
        Update internal network parameters
        :param eps_sample: (ndarray) noise sample
        :param add_eps (bool)
        :return: None
        """
        param_itr = 0
        for _param in range(len(self.params)):
            pre_param_itr = param_itr
            param_itr += self.params[_param].parameters.size
            param_sample = eps_sample[pre_param_itr:param_itr]
            self.params[_param].update_params(param_sample, add_eps=add_eps)


class Network(ESNetwork):
    def __init__(self, input_dim, output_dim, noise_std=0.01, num_eps_samples=64, sample_type="antithetic"):
        self.params = list()  # list of parameters to update

        # state and action dimensionality, defaults
        self.default_nodes = 32
        self.input_dim = input_dim
        self.output_dim = output_dim

        # meta weights
        self.eta_w = Parameter(np.zeros(1))
        self.eta_h = Parameter(np.zeros(1))
        self.final_bias = Parameter(np.zeros(output_dim))

        self.params.append(self.eta_w)
        self.params.append(self.eta_h)
        self.params.append(self.final_bias)

        change_nodes_w1 = {
            "clip":1, "activation": identity, "input_size": 1, "output_size": self.default_nodes}
        self.change_nodes_w1 = \
            NetworkModule("linear", change_nodes_w1)
        self.params.append(self.change_nodes_w1)
        change_nodes_w2 = {
            "clip":1, "activation": identity, "input_size": self.default_nodes, "output_size": 1}
        self.change_nodes_w2 = \
            NetworkModule("linear", change_nodes_w2)
        self.params.append(self.change_nodes_w2)

        # weights
        self.weight_trace1 = np.zeros((self.input_dim, self.default_nodes))
        self.weight_trace2 = np.zeros((self.default_nodes, self.output_dim))

        # weight optimized values
        self.nodes = self.default_nodes
        self.hebb_trace1 = np.zeros((self.input_dim, self.default_nodes))
        self.hebb_trace2 = np.zeros((self.default_nodes, self.output_dim))
        self.hebb_trace1 = deepcopy(self.hebb_trace1) + np.random.random(self.hebb_trace1.shape)*0.0001
        self.hebb_trace2 = deepcopy(self.hebb_trace2) + np.random.random(self.hebb_trace2.shape)*0.0001

        # templates
        self.hebb_template1 = deepcopy(self.hebb_trace1)
        self.hebb_template2 = deepcopy(self.hebb_trace2)
        self.weight_template1 = deepcopy(self.weight_trace1)
        self.weight_template2 = deepcopy(self.weight_trace2)

        super(Network, self).__init__(noise_std=noise_std,
            params=self.params, num_eps_samples=num_eps_samples, sample_type=sample_type)

    def change_nodes(self, run_reward):
        change_nodes = np.tanh(self.change_nodes_w1.forward(np.array([[run_reward]])))
        change_nodes = np.round(np.tanh(self.change_nodes_w2.forward(change_nodes)))[0][0]
        return max(min(self.nodes + change_nodes, 16), 48)

    def reset_run(self, run_reward):
        self.weight_trace1 = (1-self.eta_w.val)*self.weight_trace1 + self.eta_w.val*self.hebb_trace1
        self.weight_trace2 = (1-self.eta_w.val)*self.weight_trace2 + self.eta_w.val*self.hebb_trace2
        self.nodes = self.change_nodes(run_reward)

        self.hebb_trace1 = self.weight_trace1
        self.hebb_trace2 = self.weight_trace2

    def reset_lifetime(self):
        self.weight_trace1 = self.weight_template1
        self.weight_trace2 = self.weight_template2

    def forward(self, state):
        presynaptic = state
        post_synaptic = np.matmul(presynaptic, self.hebb_trace1)
        self.hebb_trace1 = np.clip((1 - self.eta_h.val) * self.hebb_trace1 + \
                self.eta_h.val * presynaptic.transpose()*post_synaptic, a_min=-1, a_max=1)

        presynaptic = post_synaptic
        post_synaptic = np.matmul(presynaptic, self.hebb_trace2)
        self.hebb_trace2 = np.clip((1 - self.eta_h.val) * self.hebb_trace2 + \
                self.eta_h.val * presynaptic.transpose()*post_synaptic, a_min=-1, a_max=1)

        return post_synaptic + self.final_bias.val



class EvolutionaryOptimizer:
    def __init__(self, network, num_workers=2, epsilon_samples=48, learning_rate_limit=0.001,
            environment_id="Pendulum-v0", learning_rate=0.01, weight_decay=0.01, max_iterations=2000):
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.network = network
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon_samples = epsilon_samples
        self.learning_rate_limit = learning_rate_limit
        self.optimizer = Adam(network.parameters(), learning_rate)
        self.environments = [
            gym.make(environment_id) for _ in range(num_workers)]

    def parallel_returns(self, x):
        """
        Function call for collecting parallelized rewards
        :param x: (tuple(int, int)) worker id and seed
        :return: (list) collected returns
        """
        worker_id, seed = x
        return compute_returns(seed=seed, environment=self.environments[worker_id],
            network=self.network, num_eps_samples=self.epsilon_samples//self.num_workers)

    def compute_weight_decay(self, weight_decay, model_param_list):
        """
        Compute weight decay penalty
        :param weight_decay: (float) weight decay coefficient
        :param model_param_list: (ndarray) weight parameters
        :return: (float) weight decay penalty
        """
        model_param_grid = np.array(model_param_list + self.network.parameters())
        return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

    def update(self, iteration):
        """
        Update weights of given model using OpenAI-ES
        :param iteration: (int) iteration number used for random seed sampling
        :return: None
        """
        samples = list()
        timestep_list = list()
        sample_returns = list()
        random_seeds = [(_, iteration*self.num_workers + _) for _ in range(self.num_workers)]
        with Pool(self.num_workers) as p:
            values = p.map(func=self.parallel_returns, iterable=random_seeds)

        total_timesteps = 0
        for _worker in range(self.num_workers):
            seed = random_seeds[_worker]
            returns, timesteps = values[_worker]

            timestep_list.append(timesteps)
            total_timesteps += timesteps
            sample_returns += returns.tolist()
            samples += [self.network.generate_eps_samples(
                seed[1], self.epsilon_samples//self.num_workers)]

        eps = np.concatenate(samples)
        returns = np.array(sample_returns)

        #print(np.sum(returns)/len(returns))

        # rank sort and convert rewards
        ret_len = len(returns)
        returns = [[_r, returns[_r], 0] for _r in range(ret_len)]
        returns.sort(key=lambda x: x[1], reverse=True)
        for _r in range(ret_len):
            returns[_r][2] = ((-_r + ret_len // 2) / (ret_len // 2)) / 2
        returns.sort(key=lambda x: x[0])
        returns = np.array([_r[2] for _r in returns])

        if self.weight_decay > 0:
            l2_decay = self.compute_weight_decay(self.weight_decay, eps)
            returns += l2_decay

        returns = (returns - np.mean(returns)) / (returns.std() + 1e-5)

        # compute weight update
        # sigma squared to account for sample = eps*sigma to get rid of sigma
        change_mu = (1 / (self.epsilon_samples * (self.network.es_optim.noise_std ** 2))) * np.dot(eps.T, returns)

        ratio, theta = self.optimizer.update(-change_mu)
        self.network.update_params(theta, add_eps=False)

        self.learning_rate = (1-(iteration/self.max_iterations))\
            *self.learning_rate + (iteration/self.max_iterations)*self.learning_rate_limit

        avg_return_rec = sum(sample_returns) / len(sample_returns)
        return avg_return_rec, total_timesteps



import gym

if __name__ == "__main__":
    t_time = 0.0
    #import pybullet_envs

    env_id = "CrippledHopper-v0"
    envrn = gym.make(env_id)

    envrn.reset()
    envrn.env.ES = True

    net = Network(
        envrn.observation_space.shape[0],
        envrn.action_space.shape[0],
        num_eps_samples=128,
        noise_std=0.01,
    )

    es_optim = EvolutionaryOptimizer(
        net,
        environment_id=env_id,
        num_workers=4,
        epsilon_samples=128,
        learning_rate=0.01,
        learning_rate_limit=0.001,
        max_iterations=1000
    )

    reward_list = list()
    for _i in range(es_optim.max_iterations):
        r, _ = es_optim.update(_i)
        print(r, _i)
















































