import numpy as np


class NetworkModule:
    def __init__(self, module_type, module_metadata):
        self.parameters = list()
        self.module_type = module_type
        self.module_metadata = module_metadata
        self.activation = self.module_metadata["activation"]

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

        self.param_ref_list = self.parameters
        self.parameters = np.concatenate([_p[0].flatten() for _p in self.param_ref_list])

    def reset(self):
        if self.module_type == "eligibility":
            self.hebbian_trace = self.hebbian_trace * 0
            self.eligibility_trace = self.hebbian_trace * 0

        elif self.module_type == "eligibility_recurrent":
            self.hebbian_trace = self.hebbian_trace * 0
            self.recurrent_trace = self.recurrent_trace * 0
            self.eligibility_trace = self.hebbian_trace * 0

    def update_trace(self, pre_synaptic, post_synaptic):
        if self.module_type == "eligibility":
            modulatory_signal = self.modulation_fan_out_bias + np.matmul(self.modulation_fan_out_weight,
                np.tanh(np.matmul(self.modulation_fan_in_weight, post_synaptic) + self.modulation_fan_in_bias))

            self.hebbian_trace = np.clip(
                self.hebbian_trace + modulatory_signal * self.eligibility_trace,
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)

            self.eligibility_trace = (np.ones(1) - self.eligibility_eta) * \
                self.eligibility_trace + self.eligibility_eta * (np.matmul(pre_synaptic.transpose(), post_synaptic))

        elif self.module_type == "eligibility_recurrent":
            modulatory_signal = self.modulation_fan_out_bias + np.matmul(self.modulation_fan_out_weight,
                np.tanh(np.matmul(self.modulation_fan_in_weight, post_synaptic) + self.modulation_fan_in_bias))

            self.hebbian_trace = np.clip(
                self.hebbian_trace + modulatory_signal * self.eligibility_trace,
                a_max=self.module_metadata["clip"], a_min=self.module_metadata["clip"] * -1)

            self.eligibility_trace = (np.ones(1) - self.eligibility_eta) * \
                self.eligibility_trace + self.eligibility_eta * (np.matmul(pre_synaptic.transpose(), post_synaptic))

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
            self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic)

        return post_synaptic

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




































































