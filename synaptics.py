import numpy as np

class Synapse:
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.scale = kwargs.get('scale', 1)
        self.slow_var = 0
        self.slow_tau = kwargs.get('slow_tau', 50)
        self.w = .2
        self.slow_variable_limit = kwargs.get('slow_variable_limit', None)
        if self.slow_variable_limit:
            self.slow_var_choice = self.slow_var_limited
        else:
            self.slow_var_choice = self.slow_var_unlimited


    def forward(self):
        self.postsynaptic.accumulate_current(self.presynaptic.get_output_current() * self.w)

    def pair_stdp(self, lr=.01, asymmetry=5):
        dw = 0
        if self.presynaptic.get_spike_status():
            self.w -= self.postsynaptic.get_output_current() * (1 - self.w) * lr
        if self.postsynaptic.get_spike_status():
            self.w += self.presynaptic.get_output_current() * self.w * lr * asymmetry
        self.w += dw
        if self.w > 1.:
            self.w = 1.

    '''
    Triplet-STDP related functions:
    '''
    def compute_slow_variable(self):
        if self.postsynaptic.get_spike_status():
            self.slow_var_choice()
        else:
            self.slow_var -= self.slow_var / self.slow_tau

    def slow_var_limited(self):
        self.slow_var = self.slow_variable_limit

    def slow_var_unlimited(self):
        self.slow_var += 1

    def t_stdp(self):
        self.compute_slow_variable()

    '''
    BCM rule related functions:
    '''
    def bcm(self):
        pass

    '''
    Get info:
    '''
    def get_weight(self):
        return self.w
    
    def get_slow_variable(self):
        return self.slow_var