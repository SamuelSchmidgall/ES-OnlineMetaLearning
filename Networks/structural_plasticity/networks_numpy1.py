import gym
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from Networks.utils import *
from Networks.network_modules_numpy import NetworkModule


def compute_returns(seed, environment, network, num_eps_samples, num_env_rollouts=5, penalize_act=None):
    """
    Compute average returns for a set of parameters
    :param penalize_act: (float) activation value penalty
    :param seed: (int) noise sampling seed)
    :param environment: (str) environment id for training
    :param network: (Network) network to optimize, containing parameters
    :param num_eps_samples: (int) number of noise samples
    :param num_env_rollouts: (int) number of training rollouts to average over
    :return: (np.ndarray, int) returns and total env interacts
    """
    avg_stand = 0
    returns = list()
    local_env = environment
    total_env_interacts = 0
    max_env_interacts = 1000
    network_cpy = deepcopy(network)
    eps_samples = network.generate_eps_samples(seed, num_eps_samples)
    prune_eps_samples = network.generate_prune_eps_samples(seed, num_eps_samples)

    # iterate through the noise samples
    for _sample in range(eps_samples.shape[0]):
        return_avg = 0.0
        network = deepcopy(network_cpy)
        network.update_params(eps_samples[_sample])
        network.update_prune_params(prune_eps_samples[_sample])

        # iterate through the target number of env rollouts
        for _roll in range(num_env_rollouts):
            network.reset()
            state = local_env.reset()
            # run the simulation until it terminates or max env interation terminates it
            for _inter in range(max_env_interacts):
                # forward propagate using noisy weights and plasticity values, also update trace
                state = state.reshape((1, state.size))
                action = network.forward(state)[0]
                action = np.clip(action, a_min=-1, a_max=1)
                # interact with environment
                state, reward, game_over, _ = local_env.step(action)
                if penalize_act is not None:
                    act_penalty = np.sum(np.abs(network.recur_plastic_ff2.saved_activations[-1][0]))\
                    /network.recur_plastic_ff2.saved_activations[0].shape[1]
                    reward += -penalize_act*act_penalty

                return_avg += reward
                # end sim iteration if termination state reached
                if game_over:
                    break
                avg_stand += 1
                total_env_interacts += 1
        returns.append(return_avg / num_env_rollouts)
    return np.array(returns), total_env_interacts


class ParamSampler:
    def __init__(self, sample_type, num_eps_samples, noise_std=0.01, prune_noise_std=0.001):
        """
        Evolutionary Strategies Optimizer
        :param sample_type: (str) type of noise sampling
        :param num_eps_samples: (int) number of noise samples to generate
        :param noise_std: (float) nosie standard deviation
        """
        self.noise_std = noise_std
        self.sample_type = sample_type
        self.num_eps_samples = num_eps_samples
        self.prune_noise_std = prune_noise_std
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
            epsilon_half = rand_m.randn(num_eps_samples // 2, params.size)
            sample = np.concatenate([epsilon_half, - epsilon_half]) * self.noise_std
        elif self.sample_type == "normal":
            sample = rand_m.randn(num_eps_samples, params.size) * self.noise_std

        return sample

    def prune_sample(self, params, seed, num_eps_samples):
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
            epsilon_half = rand_m.randn(num_eps_samples // 2, params.size)
            sample = np.concatenate([epsilon_half, - epsilon_half]) * self.prune_noise_std
        elif self.sample_type == "normal":
            sample = rand_m.randn(num_eps_samples, params.size) * self.prune_noise_std

        return sample


class ESNetwork:
    def __init__(self, params, prune_params, noise_std=0.01,
                 num_eps_samples=64, sample_type="antithetic", prune_noise_std=0.001):
        """
        Evolutionary Strategies network
        :param params: (np.ndarray) parameters to optimize
        :param num_eps_samples: (int) number of noise samples
        :param sample_type: (str) type of noise sampling
        """
        self.params = params
        self.prune_params = prune_params
        self.es_optim = ParamSampler(noise_std=noise_std,
            sample_type=sample_type, num_eps_samples=num_eps_samples, prune_noise_std=prune_noise_std)

    def prune_parameters(self):
        params = list()
        for _param in range(len(self.prune_params)):
            for _p_param in range(len(self.prune_params[_param])):
                params.append(self.prune_params[_param][_p_param][0])
        return np.concatenate(params, axis=0)

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

    def generate_prune_eps_samples(self, seed, num_eps_samples):
        """
        Generate noise samples for list of parameters
        :param seed: (int) random number seed
        :param num_eps_samples (int) number of noise samples to evaluate
        :return: (ndarray) parameter noise
        """
        params = self.prune_parameters()
        sample = self.es_optim.prune_sample(params, seed+1, num_eps_samples)
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

    def update_prune_params(self, eps_sample, add_eps=True):
        """
        Update internal network parameters
        :param eps_sample: (ndarray) noise sample
        :param add_eps (bool)
        :return: None
        """
        if add_eps:
            for _param in range(len(self.prune_params)):
                for _p_param in range(len(self.prune_params[_param])):
                    self.prune_params[_param][_p_param] = \
                        self.prune_params[_param][_p_param][0] + eps_sample[_p_param],\
                        self.prune_params[_param][_p_param][1]
        else:
            for _param in range(len(self.prune_params)):
                for _p_param in range(len(self.prune_params[_param])):
                    self.prune_params[_param][_p_param] = np.array([eps_sample[_p_param]]),\
                        self.prune_params[_param][_p_param][1]

class CDPNet(ESNetwork):
    def __init__(self, input_size, output_size, noise_std=0.01, prune_noise_std=0.0015,
                 action_noise_std=None, num_eps_samples=64, sample_type="antithetic"):
        """
        Eligibility Network with reward modulated plastic weights
        :param input_size: (int) size of observation space
        :param output_size: (int) size of action space
        :param num_eps_samples: (int) number of epsilon samples (population size)
        :param sample_type: (str) network noise sampling type
        """
        self.params = list()  # list of parameters to update
        self.prune_params = list()  # list of parameters to update
        self.input_size = input_size  # observation space dimensionality
        self.output_size = output_size  # action space dimensionality
        self.action_noise_std = action_noise_std  # action noise standard deviation

        recur_ff1_meta = {
            "clip": 2, "activation": identity, "input_size": input_size, "output_size": 16}
        self.recur_plastic_ff1 = \
            NetworkModule("linear", recur_ff1_meta)
        self.params.append(self.recur_plastic_ff1)

        recur_ff2_meta = {
            "clip": 2, "activation": identity, "input_size": 16, "output_size": 16}
        self.recur_plastic_ff2 = \
            NetworkModule("structural_neuromod_recurrent_eligibility", recur_ff2_meta, save_activations=True)
        self.params.append(self.recur_plastic_ff2)
        self.prune_params.append(self.recur_plastic_ff2.prune_parameters)

        recur_ff3_meta = {
            "clip": 2, "activation": identity, "input_size": 16, "output_size": output_size}
        self.recur_plastic_ff3 = \
            NetworkModule("linear", recur_ff3_meta)
        self.params.append(self.recur_plastic_ff3)

        super(CDPNet, self).__init__(noise_std=noise_std, prune_params=self.prune_params,
                params=self.params, num_eps_samples=num_eps_samples, sample_type=sample_type, prune_noise_std=prune_noise_std)

    def reset(self):
        """
        Reset inter-lifetime network parameters
        :return: None
        """
        for _param in self.params:
            _param.reset()

    def forward(self, x):
        """
        Forward propagate input value
        :param x: (ndarray) state input
        :return: (ndarray) post synaptic activity at final layer
        """

        pre_synaptic_ff1 = x
        post_synaptic_ff1 = np.tanh(
            self.recur_plastic_ff1.forward(pre_synaptic_ff1))

        pre_synaptic_ff2 = post_synaptic_ff1
        post_synaptic_ff2 = \
            self.recur_plastic_ff2.forward(pre_synaptic_ff2)

        pre_synaptic_ff3 = post_synaptic_ff2
        post_synaptic_ff3 = \
            self.recur_plastic_ff3.forward(pre_synaptic_ff3)

        if self.action_noise_std is not None:
            x += np.random.randn(*x.shape) * self.action_noise_std

        return post_synaptic_ff3


class EvolutionaryOptimizer:
    def __init__(self, network, num_workers=2, epsilon_samples=48, learning_rate_limit=0.001,
                 environment_id="Pendulum-v0", learning_rate=0.01, weight_decay=0.01, max_iterations=2000,
                 pruning_lr=0.001):
        """
        Initialization function for Natural Evolutionary Strategies (OpenAI-ES)
        :param network: (Network) numpy artificial neural network
        :param num_workers: (int) number of training instances to parallelize
        :param epsilon_samples: (int) number of noise samples per pseudo-gradient update
        :param environment_id: (str) training environment ID
        :param weight_decay: (float) penalty scale for large weight values
        :param max_iterations: (int) maximum training iterations
        :param learning_rate: (float) update learning rate
        :param learning_rate_limit: (float) learning rate decay limit
        """
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divisible by num workers"
        self.network = network
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.epsilon_samples = epsilon_samples
        self.learning_rate_limit = learning_rate_limit
        self.optimizer = Adam(network.parameters(), learning_rate)
        self.prune_optimizer = Adam(network.prune_parameters(), pruning_lr)
        self.environments = [gym.make(environment_id) for _ in range(num_workers)]

    def parallel_returns(self, x):
        """
        Function call for collecting parallelized rewards
        :param x: (tuple(int, int)) worker id and seed
        :return: (list) collected returns
        """
        worker_id, seed = x
        return compute_returns(seed=seed, environment=self.environments[worker_id],
            network=self.network, num_eps_samples=self.epsilon_samples // self.num_workers, penalize_act=0.01)

    def compute_weight_decay(self, weight_decay, model_param_list, l2=False):
        """
        Compute weight decay penalty
        :param weight_decay: (float) weight decay coefficient
        :param model_param_list: (ndarray) weight parameters
        :return: (float) weight decay penalty
        """
        if not l2:
            model_param_grid = np.array(model_param_list + self.network.parameters())
            return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)
        elif l2:
            model_param_grid = np.power(np.array(model_param_list + self.network.parameters()), 2)
            return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

    def compute_prune_decay(self, weight_decay, model_param_list):
        """
        Compute weight decay penalty
        :param weight_decay: (float) weight decay coefficient
        :param model_param_list: (ndarray) weight parameters
        :return: (float) weight decay penalty
        """
        model_param_grid = np.array(model_param_list + self.network.prune_parameters())
        return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

    def update(self, iteration):
        """
        Update weights of given model using OpenAI-ES
        :param iteration: (int) iteration number used for random seed sampling
        :return: None
        """
        samples = list()
        prune_samples = list()
        timestep_list = list()
        sample_returns = list()
        random_seeds = [
            (_, iteration * self.num_workers + _) for _ in range(self.num_workers)]
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
                seed[1], self.epsilon_samples // self.num_workers)]
            prune_samples += [self.network.generate_prune_eps_samples(
                seed[1], self.epsilon_samples // self.num_workers)]

        eps = np.concatenate(samples)
        prune_eps = np.concatenate(prune_samples)
        returns = np.array(sample_returns)

        # rank sort and convert rewards
        ret_len = len(returns)
        returns = [[_r, returns[_r], 0] for _r in range(ret_len)]
        returns.sort(key=lambda x: x[1], reverse=True)
        for _r in range(ret_len):
            returns[_r][2] = ((-_r + ret_len // 2) / (ret_len // 2)) / 2
        returns.sort(key=lambda x: x[0])
        returns = np.array([_r[2] for _r in returns])

        # weight decay functionality
        if self.weight_decay > 0:
            l2_decay = self.compute_weight_decay(self.weight_decay, eps, l2=True)
            returns += l2_decay

        ## weight decay functionality
        #if self.weight_decay > 0:
        #    l2_decay = self.compute_prune_decay(self.weight_decay, eps)
        #    returns += l2_decay

        # normalize returns
        returns = (returns - np.mean(returns)) / (returns.std() + 1e-5)

        # compute weight update
        # sigma squared to account for sample = eps*sigma to get rid of sigma
        change_mu = (1 / (self.epsilon_samples * (self.network.es_optim.noise_std ** 2))) * np.dot(eps.T, returns)
        change_mu_prune = (1 / (self.epsilon_samples * (self.network.es_optim.noise_std ** 2))) * np.dot(prune_eps.T, returns)

        ratio, theta = self.optimizer.update(-change_mu)
        prune_ratio, prune_theta = self.prune_optimizer.update(-change_mu_prune)

        self.network.update_params(theta, add_eps=False)
        self.network.update_prune_params(prune_theta, add_eps=False)

        # update learning rate
        self.learning_rate = (1 - (iteration / self.max_iterations)) \
            * self.learning_rate + (iteration / self.max_iterations) * self.learning_rate_limit

        # compute return average
        avg_return_rec = sum(sample_returns) / len(sample_returns)
        return avg_return_rec, total_timesteps


if __name__ == "__main__":
    t_time = 0.0
    # import pybullet_envs

    env_id = "InvertedDoublePendulum-v2"
    envrn = gym.make(env_id)

    envrn.reset()
    envrn.env.ES = True

    """ ~~~ HYPER PARAMETERS ~~~ """
    lr = 0.01
    lr_lim = 0.001
    w_decay = 0.5
    prune_lr = 0.001

    eps_noise = 0.015
    act_noise = 0.0015
    prune_eps_noise = 0.015

    workers = 2
    iterations = 1000
    eps_samples = 48*4
    """ ~~~~~~~~~~~~~~~~~~~~~~~~ """

    net = CDPNet(
        envrn.observation_space.shape[0],
        envrn.action_space.shape[0],
        action_noise_std=act_noise,
        num_eps_samples=eps_samples,
        noise_std=eps_noise,
        prune_noise_std=prune_eps_noise
    )

    es_optim = EvolutionaryOptimizer(
        network=net,
        environment_id=env_id,
        num_workers=workers,
        epsilon_samples=eps_samples,
        learning_rate=lr,
        learning_rate_limit=lr_lim,
        pruning_lr=prune_lr,
        max_iterations=iterations,
        weight_decay=w_decay
    )

    import pickle

    reward_list = list()
    for _i in range(es_optim.max_iterations):
        r, t = es_optim.update(_i)
        t_time += t
        print(r, _i, t / 48, t_time)
        reward_list.append((r, _i, t_time))
        if _i%20 == 0:
            with open("model_sp2.pkl", "wb") as f:
                pickle.dump(net, f)
            with open("reward_sp2.pkl", "wb") as f:
                pickle.dump(reward_list, f)
