import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Parameter:
    def __init__(self, var):
        self.val = var
        self.parameters = var

    def params(self):
        return self.parameters

    def update_params(self, eps, add_eps=True):
        self.parameters = self.parameters + eps
        self.val = self.parameters


class NetworkModule:
    def __init__(self, module_type, module_metadata, save_activations=False):
        self.parameters = list()
        self.prune_parameters = list()
        self.module_type = module_type
        self.module_metadata = module_metadata
        self.activation = self.module_metadata["activation"]

        self.saved_activations = list()
        self.save_activations = save_activations

        if self.module_type == "linear":
            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))

            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

        elif self.module_type == "eligibility":
            self.recurrent_trace = np.zeros((1, module_metadata["output_size"]))
            self.hebbian_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.eligibility_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))

            self.eligibility_eta = np.zeros(1)
            self.parameters.append((self.eligibility_eta, "eligibility_eta"))

            self.modulation_fan_in_weight = np.zeros((module_metadata["output_size"], 1))
            self.parameters.append((self.modulation_fan_in_weight, "modulation_fan_in_weight"))

            self.modulation_fan_in_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_in_bias, "modulation_fan_in_bias"))

            self.modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_out_weight, "modulation_fan_out_weight"))

            self.modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.modulation_fan_out_bias, "modulation_fan_out_bias"))

            self.alpha_plasticity = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.parameters.append((self.alpha_plasticity, "alpha_plasticity"))

            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))

            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

        elif self.module_type == "eligibility_recurrent":
            self.recurrent_trace = np.zeros((1, module_metadata["output_size"]))
            self.hebbian_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.eligibility_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))

            self.eligibility_eta = np.zeros(1)
            self.parameters.append((self.eligibility_eta, "eligibility_eta"))

            self.modulation_fan_in_weight = np.zeros((module_metadata["output_size"], 1))
            self.parameters.append((self.modulation_fan_in_weight, "modulation_fan_in_weight"))

            self.modulation_fan_in_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_in_bias, "modulation_fan_in_bias"))

            self.modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_out_weight, "modulation_fan_out_weight"))

            self.modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.modulation_fan_out_bias, "modulation_fan_out_bias"))

            self.alpha_plasticity = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.parameters.append((self.alpha_plasticity, "alpha_plasticity"))

            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))

            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

            self.recurrent_layer_weight = np.zeros((
                module_metadata["output_size"], module_metadata["output_size"]))
            self.parameters.append((self.recurrent_layer_weight, "recurrent_layer_weight"))

            self.recurrent_layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.recurrent_layer_bias, "recurrent_layer_bias"))

        elif self.module_type == 'simple_neuromod_recurrent':
            self.recurrent_trace = np.zeros((1, module_metadata["output_size"]))
            self.hebbian_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))

            self.modulation_fan_in_weight = np.zeros((module_metadata["output_size"], 1))
            self.parameters.append((self.modulation_fan_in_weight, "modulation_fan_in_weight"))

            self.modulation_fan_in_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_in_bias, "modulation_fan_in_bias"))

            self.modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_out_weight, "modulation_fan_out_weight"))

            self.modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.modulation_fan_out_bias, "modulation_fan_out_bias"))

            self.alpha_plasticity = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.parameters.append((self.alpha_plasticity, "alpha_plasticity"))

            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))

            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

            self.recurrent_layer_weight = np.zeros((
                module_metadata["output_size"], module_metadata["output_size"]))
            self.parameters.append((self.recurrent_layer_weight, "recurrent_layer_weight"))

            self.recurrent_layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.recurrent_layer_bias, "recurrent_layer_bias"))

        elif self.module_type == 'simple_neuromod':
            self.hebbian_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))

            self.modulation_fan_in_weight = np.zeros((module_metadata["output_size"], 1))
            self.parameters.append((self.modulation_fan_in_weight, "modulation_fan_in_weight"))

            self.modulation_fan_in_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_in_bias, "modulation_fan_in_bias"))

            self.modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_out_weight, "modulation_fan_out_weight"))

            self.modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.modulation_fan_out_bias, "modulation_fan_out_bias"))

            self.alpha_plasticity = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.parameters.append((self.alpha_plasticity, "alpha_plasticity"))

            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))

            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

        elif self.module_type == 'structural_neuromod_recurrent_eligibility':

            self.linear_eligibility_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.linear_hebbian_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.linear_eligibility_eta = np.zeros(1)
            self.parameters.append((self.linear_eligibility_eta, "linear_eligibility_eta"))
            self.prune_parameters.append((np.zeros(1), "linear_prune_param"))
            self.linear_modulation_fan_in_weight = np.zeros((module_metadata["input_size"], 1))
            self.parameters.append((self.linear_modulation_fan_in_weight, "linear_modulation_fan_in_weight"))
            self.linear_modulation_fan_in_bias = np.zeros((1, module_metadata["input_size"]))
            self.parameters.append((self.linear_modulation_fan_in_bias, "linear_modulation_fan_in_bias"))
            self.linear_modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.linear_modulation_fan_out_weight, "linear_modulation_fan_out_weight"))
            self.linear_modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.linear_modulation_fan_out_bias, "linear_modulation_fan_out_bias"))
            self.linear_alpha_plasticity = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.parameters.append((self.linear_alpha_plasticity, "linear_alpha_plasticity"))
            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))
            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

            self.recurrent_eligibility_trace = np.zeros((
                (module_metadata["output_size"], module_metadata["output_size"])))
            self.recurrent_hebbian_trace = np.zeros((
                (module_metadata["output_size"], module_metadata["output_size"])))
            self.recurrent_eligibility_eta = np.zeros(1)
            self.parameters.append((self.recurrent_eligibility_eta, "recurrent_eligibility_eta"))
            self.prune_parameters.append((np.zeros(1), "recurrent_prune_param"))
            self.recurrent_modulation_fan_in_weight = np.zeros((module_metadata["output_size"], 1))
            self.parameters.append((self.recurrent_modulation_fan_in_weight, "recurrent_modulation_fan_in_weight"))
            self.recurrent_modulation_fan_in_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.recurrent_modulation_fan_in_bias, "recurrent_modulation_fan_in_bias"))
            self.recurrent_modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.recurrent_modulation_fan_out_weight, "recurrent_modulation_fan_out_weight"))
            self.recurrent_modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.recurrent_modulation_fan_out_bias, "recurrent_modulation_fan_out_bias"))
            self.recurrent_alpha_plasticity = np.zeros((
                (module_metadata["output_size"], module_metadata["output_size"])))
            self.parameters.append((self.recurrent_alpha_plasticity, "recurrent_alpha_plasticity"))
            self.recurrent_trace = np.zeros(
                (1, module_metadata["output_size"]))
            self.recurrent_layer_weight = np.zeros((
                module_metadata["output_size"], module_metadata["output_size"]))
            self.parameters.append((self.recurrent_layer_weight, "recurrent_layer_weight"))
            self.recurrent_layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.recurrent_layer_bias, "recurrent_layer_bias"))

        elif self.module_type == 'structural_simple_neuromod':
            self.hebbian_trace = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))

            #self.prune_param = np.zeros(1)
            self.prune_parameters.append((np.zeros(1), "prune_param"))

            self.modulation_fan_in_weight = np.zeros((module_metadata["output_size"], 1))
            self.parameters.append((self.modulation_fan_in_weight, "modulation_fan_in_weight"))

            self.modulation_fan_in_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_in_bias, "modulation_fan_in_bias"))

            self.modulation_fan_out_weight = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.modulation_fan_out_weight, "modulation_fan_out_weight"))

            self.modulation_fan_out_bias = np.zeros((1, 1))
            self.parameters.append((self.modulation_fan_out_bias, "modulation_fan_out_bias"))

            self.alpha_plasticity = np.zeros((
                (module_metadata["input_size"], module_metadata["output_size"])))
            self.parameters.append((self.alpha_plasticity, "alpha_plasticity"))

            self.layer_weight = np.zeros(
                (module_metadata["input_size"], module_metadata["output_size"]))
            self.parameters.append((self.layer_weight, "layer_weight"))

            self.layer_bias = np.zeros((1, module_metadata["output_size"]))
            self.parameters.append((self.layer_bias, "layer_bias"))

        self.param_ref_list = self.parameters
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])
        if len(self.prune_parameters) > 0:
            self.prune_parameters = self.prune_parameters

    def reset(self):
        if self.module_type == "eligibility":
            self.hebbian_trace = self.hebbian_trace * 0
            self.eligibility_trace = self.hebbian_trace * 0

        elif self.module_type in ["eligibility_recurrent"]:
            self.hebbian_trace = self.hebbian_trace * 0
            self.recurrent_trace = self.recurrent_trace * 0
            self.eligibility_trace = self.hebbian_trace * 0

        elif self.module_type == 'simple_neuromod_recurrent':
            self.hebbian_trace = self.hebbian_trace * 0
            self.recurrent_trace = self.recurrent_trace * 0

        elif self.module_type in ["structural_neuromod_recurrent_eligibility"]:
            self.linear_hebbian_trace = self.linear_hebbian_trace * 0
            self.linear_eligibility_trace = self.linear_hebbian_trace * 0

            self.recurrent_trace = self.recurrent_trace * 0
            self.recurrent_hebbian_trace = self.recurrent_hebbian_trace * 0
            self.recurrent_eligibility_trace = self.recurrent_hebbian_trace * 0

        elif self.module_type in ['simple_neuromod', 'structural_simple_neuromod']:
            self.hebbian_trace = self.hebbian_trace * 0

        if self.save_activations:
            self.saved_activations.clear()

    def update_trace(self, pre_synaptic, post_synaptic):
        if self.module_type == "eligibility":
            modulatory_signal = self.modulation_fan_out_bias + np.matmul(self.modulation_fan_out_weight,
                np.tanh(np.matmul(self.modulation_fan_in_weight, post_synaptic) + self.modulation_fan_in_bias))
            self.hebbian_trace = np.clip(
                self.hebbian_trace + modulatory_signal * self.eligibility_trace,
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)
            self.eligibility_trace = (np.ones(1) - self.eligibility_eta) * \
                self.eligibility_trace + self.eligibility_eta * (np.matmul(pre_synaptic.transpose(), post_synaptic))

        elif self.module_type in ["eligibility_recurrent"]:
            modulatory_signal = self.modulation_fan_out_bias + np.matmul(self.modulation_fan_out_weight,
                np.tanh(np.matmul(self.modulation_fan_in_weight, post_synaptic) + self.modulation_fan_in_bias))
            self.hebbian_trace = np.clip(
                self.hebbian_trace + modulatory_signal * self.eligibility_trace,
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)
            self.eligibility_trace = (np.ones(1) - self.eligibility_eta) * \
                self.eligibility_trace + self.eligibility_eta * (np.matmul(pre_synaptic.transpose(), post_synaptic))

        elif self.module_type in ["structural_neuromod_recurrent_eligibility"]:
            linear_pre_synaptic, recurrent_pre_synaptic = pre_synaptic
            linear_post_synaptic, recurrent_post_synaptic = post_synaptic

            linear_modulatory_signal = self.linear_modulation_fan_out_bias + np.matmul(self.linear_modulation_fan_out_weight,
                np.tanh(np.matmul(self.linear_modulation_fan_in_weight, linear_pre_synaptic) + self.linear_modulation_fan_in_bias))
            self.linear_hebbian_trace = np.clip(
                self.linear_hebbian_trace + linear_modulatory_signal * self.linear_eligibility_trace,
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)
            self.linear_eligibility_trace = np.clip((np.ones(1) - self.linear_eligibility_eta) * \
                self.linear_eligibility_trace + self.linear_eligibility_eta * \
                        (np.matmul(linear_pre_synaptic.transpose(), linear_post_synaptic)),
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)

            recurrent_modulatory_signal = self.recurrent_modulation_fan_out_bias + np.matmul(self.recurrent_modulation_fan_out_weight,
                np.tanh(np.matmul(self.recurrent_modulation_fan_in_weight, recurrent_pre_synaptic) + self.recurrent_modulation_fan_in_bias))
            self.recurrent_hebbian_trace = np.clip(
                self.recurrent_hebbian_trace + recurrent_modulatory_signal * self.recurrent_eligibility_trace,
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)
            self.recurrent_eligibility_trace = np.clip((np.ones(1) - self.recurrent_eligibility_eta) * \
                self.recurrent_eligibility_trace + self.recurrent_eligibility_eta * \
                        (np.matmul(recurrent_pre_synaptic.transpose(), recurrent_post_synaptic)),
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)

        elif self.module_type in ['simple_neuromod_recurrent', 'simple_neuromod', 'structural_simple_neuromod']:
            modulatory_signal = self.modulation_fan_out_bias + \
                np.matmul(self.modulation_fan_out_weight, np.tanh(
                    np.matmul(self.modulation_fan_in_weight, post_synaptic) + self.modulation_fan_in_bias))
            self.hebbian_trace = np.clip(
                self.hebbian_trace + modulatory_signal * (np.matmul(pre_synaptic.transpose(), post_synaptic)),
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)

    def forward(self, x):
        post_synaptic = None
        pre_synaptic = x.copy()
        if self.module_type == "linear":
            post_synaptic = self.activation(np.matmul(x, self.layer_weight) + self.layer_bias)

        elif self.module_type == "eligibility":
            fixed_weights = np.matmul(x, self.layer_weight) + self.layer_bias
            plastic_weights = np.matmul(x, (self.alpha_plasticity * self.hebbian_trace))
            post_synaptic = self.activation(fixed_weights + plastic_weights)
            self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

        elif self.module_type == "eligibility_recurrent":
            # todo: modulated recurrence
            fixed_ff_weights = np.matmul(x, self.layer_weight) + self.layer_bias
            plastic_ff_weights = np.matmul(x, (self.alpha_plasticity * self.hebbian_trace))
            fixed_rec_weights = \
                np.matmul(self.recurrent_trace, self.recurrent_layer_weight) + self.recurrent_layer_bias
            post_synaptic = self.activation(fixed_ff_weights + plastic_ff_weights + fixed_rec_weights)
            self.recurrent_trace = post_synaptic
            self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

        elif self.module_type == 'simple_neuromod_recurrent':
            fixed_ff_weights = np.matmul(x, self.layer_weight) + self.layer_bias
            plastic_ff_weights = np.matmul(x, (self.alpha_plasticity * self.hebbian_trace))
            fixed_rec_weights = \
                np.matmul(self.recurrent_trace, self.recurrent_layer_weight) + self.recurrent_layer_bias
            post_synaptic = self.activation(fixed_ff_weights + plastic_ff_weights + fixed_rec_weights)
            self.recurrent_trace = post_synaptic
            self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

        elif self.module_type == 'simple_neuromod':
            fixed_ff_weights = np.matmul(x, self.layer_weight) + self.layer_bias
            plastic_ff_weights = np.matmul(x, (self.alpha_plasticity * self.hebbian_trace))
            post_synaptic = self.activation(fixed_ff_weights + plastic_ff_weights)
            self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

        elif self.module_type == 'structural_simple_neuromod':
            weights = self.layer_weight + self.alpha_plasticity * self.hebbian_trace
            s_plast_weights = self.structural_plast(weights, self.prune_parameters[0][0])
            #a = np.sum(np.where(s_plast_weights != 0, 0, 1))
            ff_weights = np.matmul(x, s_plast_weights) + self.layer_bias
            post_synaptic = self.activation(ff_weights)
            self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

        elif self.module_type in ["structural_neuromod_recurrent_eligibility"]:
            fixed_ff_weights = self.layer_weight
            plastic_ff_weights = self.linear_alpha_plasticity * self.linear_hebbian_trace
            ff_weights = self.structural_plast(fixed_ff_weights + plastic_ff_weights, self.prune_parameters[0][0])
            ff_post_syn = np.matmul(x, ff_weights) + self.layer_bias
            #a = np.sum(np.where(ff_weights != 0, 0, 1))

            fixed_rec_weights = self.recurrent_layer_weight
            plastic_rec_weights = self.recurrent_alpha_plasticity * self.recurrent_hebbian_trace
            rec_weights = self.structural_plast(fixed_rec_weights + plastic_rec_weights, self.prune_parameters[1][0])
            rec_post_syn = np.matmul(self.recurrent_trace, rec_weights) + self.recurrent_layer_bias
            #b = np.sum(np.where(rec_weights != 0, 0, 1))

            post_synaptic = self.activation(ff_post_syn + rec_post_syn)
            self.recurrent_trace = post_synaptic
            self.update_trace(
                pre_synaptic=(pre_synaptic, self.recurrent_trace), post_synaptic=(post_synaptic, post_synaptic))

        # potentially used as an activity penalty
        if self.save_activations:
            self.saved_activations.append(post_synaptic)

        return post_synaptic

    @staticmethod
    def structural_plast(w, p_param):
        return np.where(np.abs(w) < np.abs(np.tanh(p_param)), 0, w)

    def params(self):
        return self.parameters

    def update_params(self, eps, add_eps=True):
        eps_index = 0
        for _ref in range(len(self.param_ref_list)):
            _val, _str_ref = self.param_ref_list[_ref]
            pre_eps = eps_index
            eps_index = eps_index + _val.size
            if add_eps:
                new_val = _val.flatten() + eps[pre_eps:eps_index]
            else:
                new_val = eps[pre_eps:eps_index]
            new_val = new_val.reshape(self.param_ref_list[_ref][0].shape)

            self.param_ref_list[_ref] = new_val, _str_ref
            setattr(self, _str_ref, new_val)
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])




































































