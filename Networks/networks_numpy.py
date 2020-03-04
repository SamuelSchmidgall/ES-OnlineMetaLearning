import gym
import numpy as np
from copy import deepcopy
from multiprocessing import Pool
from Networks.utils import *
from Networks.network_modules_numpy import NetworkModule


def compute_returns(seed, environment, network, num_eps_samples, num_env_rollouts=4):
    """

    :param seed:
    :param environment:
    :param network:
    :param num_eps_samples:
    :param num_env_rollouts:
    :return:
    """
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
                return_avg += reward
                # end sim iteration if termination state reached
                if game_over:
                    break
                avg_stand += 1
                total_env_interacts += 1
        returns.append(return_avg / num_env_rollouts)
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
        """

        :param params:
        :param num_eps_samples:
        :param sample_type:
        """
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


class SpinalCPG(ESNetwork):
    def __init__(self, input_size, output_size, noise_std=0.01,
            action_noise_std=None, num_eps_samples=64, sample_type="antithetic"):
        """
        
        :param input_size:
        :param output_size:
        :param action_noise_std:
        :param num_eps_samples:
        :param sample_type:
        """
        self.params = list()
        self.input_size = input_size
        self.output_size = output_size
        self.connectivity_type = "linear"  # eligibility
        self.action_noise_std = action_noise_std

        inp_cpg_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size, "output_size": 32}
        self.inp_cpg = \
            NetworkModule("linear", inp_cpg_meta)
        self.params.append(self.inp_cpg)

        cpg_osc1_meta = {
            "clip":1, "activation": identity,
            "input_size": 32, "output_size": 32}
        self.cpg_osc_cluster1_1 = \
            NetworkModule("linear", cpg_osc1_meta)
        self.params.append(self.cpg_osc_cluster1_1)

        cpg_osc2_meta = {
            "clip":1, "activation": identity,
            "input_size": 32, "output_size": 32}
        self.cpg_osc_cluster1_2 = \
            NetworkModule("linear", cpg_osc2_meta)
        self.params.append(self.cpg_osc_cluster1_2)

        cpg_leg1_meta = {
            "clip":1, "activation": identity,
            "input_size": 32, "output_size": output_size//2}
        self.cpg_osc_cluster1_leg1 = \
            NetworkModule("linear", cpg_leg1_meta)
        self.params.append(self.cpg_osc_cluster1_leg1)

        cpg_leg2_meta = {
            "clip":1, "activation": identity,
            "input_size": 32, "output_size": output_size//2}
        self.cpg_osc_cluster1_leg2 = \
            NetworkModule("linear", cpg_leg2_meta)
        self.params.append(self.cpg_osc_cluster1_leg2)

        super(SpinalCPG, self).__init__(
            params=self.params, num_eps_samples=num_eps_samples, sample_type=sample_type)

        self.layer_1_prev = np.zeros((1, 32))
        self.layer_2_prev = np.zeros((1, 32))

    def reset(self):
        for _param in self.params:
            _param.reset()
        self.layer_1_prev = self.layer_1_prev * 0
        self.layer_2_prev = self.layer_2_prev * 0

    def forward(self, x):
        cpg_latent = self.inp_cpg.forward(x)
        cpg_latent1 = np.tanh(self.cpg_osc_cluster1_2.forward(self.layer_2_prev) + cpg_latent)
        cpg_latent2 = np.tanh(self.cpg_osc_cluster1_1.forward(self.layer_1_prev) + cpg_latent)
        self.layer_1_prev = deepcopy(cpg_latent1)
        self.layer_2_prev = deepcopy(cpg_latent2)
        leg1 = self.cpg_osc_cluster1_leg1.forward(cpg_latent1)
        leg2 = self.cpg_osc_cluster1_leg2.forward(cpg_latent2)
        return np.concatenate(leg1, leg2, axis=0)


class SpinalNetworkES(ESNetwork):
    def __init__(self, input_size, output_size, upstream_dim=8,
            action_noise_std=None, num_eps_samples=64, sample_type="antithetic"):
        """
        Spinal Network with lateral, residual upstream, residual downstream,
         recurrence and eligibility modulated weights for each set of connections
        :param input_size: (int) size of observation space
        :param output_size: (int) size of action space
        :param upstream_dim: (int) upsteam bottleneck dimensionality
        :param num_eps_samples: (int) number of epsilon samples (population size)
        :param sample_type: (str) network noise sampling type
        """
        self.params = list()
        self.input_size = input_size
        self.output_size = output_size
        self.res_connectivity_type = "linear"
        self.action_noise_std = action_noise_std
        self.input_upstream_dim = upstream_dim*2
        self.ff_connectivity_type = "linear"  # eligibility
        self.res_connectivity_type = "linear"  # eligibility

        recur_ff1_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size + self.input_upstream_dim, "output_size": 32}
        self.recur_plastic_ff1 = \
            NetworkModule(self.ff_connectivity_type, recur_ff1_meta)
        self.params.append(self.recur_plastic_ff1)
        recur_ff2_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": 32}
        self.recur_plastic_ff2 = \
            NetworkModule(self.ff_connectivity_type, recur_ff2_meta)
        self.params.append(self.recur_plastic_ff2)
        recur_ff3_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": output_size}
        self.recur_plastic_ff3 = \
            NetworkModule(self.ff_connectivity_type, recur_ff3_meta)
        self.params.append(self.recur_plastic_ff3)

        resid_down_inp_2_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size+self.input_upstream_dim, "output_size": 32}
        self.resid_downstream_inp_2 = \
            NetworkModule(self.res_connectivity_type, resid_down_inp_2_meta)
        self.params.append(self.resid_downstream_inp_2)
        resid_down_inp_3_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size+self.input_upstream_dim, "output_size": output_size}
        self.resid_downstream_inp_3 = \
            NetworkModule(self.res_connectivity_type, resid_down_inp_3_meta)
        self.params.append(self.resid_downstream_inp_3)
        resid_down_1_3_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": output_size}
        self.resid_downstream_1_3 = \
            NetworkModule(self.res_connectivity_type, resid_down_1_3_meta)
        self.params.append(self.resid_downstream_1_3)

        resid_up_2_inp_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": upstream_dim}
        self.resid_upstream_2_inp = \
            NetworkModule(self.res_connectivity_type, resid_up_2_inp_meta)
        self.params.append(self.resid_upstream_2_inp)
        resid_up_3_inp_meta = {
            "clip":1, "activation": identity, "input_size": output_size, "output_size": upstream_dim}
        self.resid_upstream_3_inp = \
            NetworkModule(self.res_connectivity_type, resid_up_3_inp_meta)
        self.params.append(self.resid_upstream_3_inp)
        resid_up_3_1_meta = {
            "clip":1, "activation": identity, "input_size": output_size, "output_size": 32}
        self.resid_upstream_3_1 = \
            NetworkModule(self.res_connectivity_type, resid_up_3_1_meta)
        self.params.append(self.resid_upstream_3_1)

        resid_lat_1_inp_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": 32}
        self.resid_lat_1 = \
            NetworkModule(self.res_connectivity_type, resid_lat_1_inp_meta)
        self.params.append(self.resid_lat_1)
        resid_lat_2_inp_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": output_size}
        self.resid_lat_2 = \
            NetworkModule(self.res_connectivity_type, resid_lat_2_inp_meta)
        self.params.append(self.resid_lat_2)
        resid_lat_3_inp_meta = {
            "clip":1, "activation": identity, "input_size": output_size, "output_size": output_size}
        self.resid_lat_3 = \
            NetworkModule(self.res_connectivity_type, resid_lat_3_inp_meta)
        self.params.append(self.resid_lat_3)

        super(SpinalNetworkES, self).__init__(
            params=self.params, num_eps_samples=num_eps_samples, sample_type=sample_type)

        self.layer_1_prev = np.zeros((1, recur_ff1_meta["output_size"]))
        self.layer_2_prev = np.zeros((1, recur_ff2_meta["output_size"]))
        self.layer_3_prev = np.zeros((1, recur_ff3_meta["output_size"]))

    def reset(self):
        """
        Reset inter-lifetime network parameters
        :return: None
        """
        for _param in self.params:
            _param.reset()
        self.layer_1_prev = self.layer_1_prev * 0
        self.layer_2_prev = self.layer_2_prev * 0
        self.layer_3_prev = self.layer_3_prev * 0

    def forward(self, x):
        """
        Forward propagate input value
        :param x: (ndarray) state input
        :return: (ndarray) post synaptic activity at final layer
        """
        upstr_res_inp_2 = self.resid_upstream_2_inp.forward(self.layer_2_prev)
        upstr_res_inp_3 = self.resid_upstream_3_inp.forward(self.layer_3_prev)
        x = np.concatenate([x, upstr_res_inp_2, upstr_res_inp_3], axis=1)

        pre_synaptic_ff1 = x
        lat_res1 = self.resid_lat_1.forward(self.layer_1_prev)
        upstr_res_3_1 = self.resid_upstream_3_1.forward(self.layer_3_prev)
        post_synaptic_ff1 = np.tanh(
            self.recur_plastic_ff1.forward(pre_synaptic_ff1) + upstr_res_3_1 + lat_res1)

        pre_synaptic_ff2 = post_synaptic_ff1
        lat_res2 = self.resid_lat_2.forward(self.layer_2_prev)
        downstr_res_inp_2 = self.resid_downstream_inp_2.forward(x)
        post_synaptic_ff2 = np.tanh(
            self.recur_plastic_ff2.forward(pre_synaptic_ff2) + downstr_res_inp_2 + lat_res2)

        pre_synaptic_ff3 = post_synaptic_ff2
        lat_res3 = self.resid_lat_3.forward(self.layer_3_prev)
        downstr_res_inp_3 = self.resid_downstream_inp_3.forward(x)
        downstr_res_1_3 = self.resid_downstream_1_3.forward(post_synaptic_ff1)
        post_synaptic_ff3 = \
            self.recur_plastic_ff3.forward(pre_synaptic_ff3) + downstr_res_1_3 + downstr_res_inp_3 + lat_res3

        self.layer_1_prev = post_synaptic_ff1
        self.layer_2_prev = post_synaptic_ff2
        self.layer_3_prev = post_synaptic_ff3


        if self.action_noise_std is not None:
            x += np.random.randn(*x.shape)*self.action_noise_std

        return post_synaptic_ff3



class EligibilityModulatedNet(ESNetwork):
    def __init__(self, input_size, output_size, noise_std=0.01,
            action_noise_std=None, num_eps_samples=64, sample_type="antithetic"):
        """
        Eligibility Network with reward modulated plastic weights
        :param input_size: (int) size of observation space
        :param output_size: (int) size of action space
        :param num_eps_samples: (int) number of epsilon samples (population size)
        :param sample_type: (str) network noise sampling type
        """
        self.params = list()  # list of parameters to update
        self.input_size = input_size  # observation space dimensionality
        self.output_size = output_size  # action space dimensionality
        self.action_noise_std = action_noise_std  # action noise standard deviation
        self.ff_connectivity_type = "linear" #"eligibility"  # connectivity type -- eligibility

        recur_ff1_meta = {
            "clip":1, "activation": identity, "input_size": input_size, "output_size": 64}
        self.recur_plastic_ff1 = \
            NetworkModule(self.ff_connectivity_type, recur_ff1_meta)
        self.params.append(self.recur_plastic_ff1)
        recur_ff2_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": 32}
        self.recur_plastic_ff2 = \
            NetworkModule(self.ff_connectivity_type, recur_ff2_meta)
        self.params.append(self.recur_plastic_ff2)
        recur_ff3_meta = {
            "clip":1, "activation": identity, "input_size": 32, "output_size": output_size}
        self.recur_plastic_ff3 = \
            NetworkModule(self.ff_connectivity_type, recur_ff3_meta)
        self.params.append(self.recur_plastic_ff3)

        super(EligibilityModulatedNet, self).__init__(noise_std=noise_std,
            params=self.params, num_eps_samples=num_eps_samples, sample_type=sample_type)

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
        post_synaptic_ff2 = np.tanh(
            self.recur_plastic_ff2.forward(pre_synaptic_ff2))

        pre_synaptic_ff3 = post_synaptic_ff2
        post_synaptic_ff3 = \
            self.recur_plastic_ff3.forward(pre_synaptic_ff3)

        if self.action_noise_std is not None:
            x += np.random.randn(*x.shape)*self.action_noise_std

        return post_synaptic_ff3



class EvolutionaryOptimizer:
    def __init__(self, network, num_workers=2, epsilon_samples=48,
            environment_id="Pendulum-v0", learning_rate=0.001, weight_decay=0.01):
        """

        :param network:
        :param num_workers:
        :param epsilon_samples:
        :param environment_id:
        :param learning_rate:
        """
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.network = network
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.epsilon_samples = epsilon_samples
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

        avg_return_rec = sum(sample_returns) / len(sample_returns)
        print(avg_return_rec)


#import pybullet_envs
#import rex_gym
#
#env_id = "RexWalk-v0"
#envrn = gym.make(env_id)
#envrn.reset()
#
#spinal_net = EligibilityModulatedNet(
#    envrn.observation_space.shape[0], envrn.action_space.shape[0], action_noise_std=0.001)
#
#es_optim = EvolutionaryOptimizer(
#    spinal_net, environment_id=env_id,
#    learning_rate=0.01, epsilon_samples=128, num_workers=16)
#
#import pickle
#
#for _i in range(2000):
#    if _i%10==0:
#        with open("save_ESnetTURN.pkl", "wb") as f:
#            pickle.dump(es_optim, f)
#    es_optim.update(_i)

















