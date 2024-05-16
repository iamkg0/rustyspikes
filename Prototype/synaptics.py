import numpy as np

class Synapse:
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.scale = kwargs.get('scale', 1)
        self.slow_var = 0
        self.slow_tau = kwargs.get('slow_tau', 100)
        self.forget_tau = kwargs.get('forget_tau', 10000)
        #self.w = np.random.uniform(0, 1)
        self.w = 1
        self.slow_variable_limit = kwargs.get('slow_variable_limit', False)
        if self.slow_variable_limit:
            self.slow_var_choice = self.slow_var_limited
        else:
            self.slow_var_choice = self.slow_var_unlimited


    def forward(self):
        self.postsynaptic.accumulate_current(self.presynaptic.get_output_current() * self.w * self.scale)

    def forgetting(self):
        dw = -(self.w * self.postsynaptic.get_output_current()) / self.forget_tau
        self.w += dw

    '''
    Pair-based STDP rule:
    '''
    def pair_stdp(self, lr=.01, asymmetry=5):
        dw = 0
        if self.presynaptic.get_spike_status():
            dw -= self.postsynaptic.get_output_current() * self.w * lr * asymmetry
        if self.postsynaptic.get_spike_status():
            dw += self.presynaptic.get_output_current() * (1 - self.w) * lr
        self.w += dw

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

    def t_stdp(self, lr=.01, asymmetry=1):
        self.compute_slow_variable()
        dw = 0
        if self.presynaptic.get_spike_status():
            dw -= self.postsynaptic.get_output_current() * self.w * lr * asymmetry
        if self.postsynaptic.get_spike_status():
            dw += self.presynaptic.get_output_current() * self.slow_var * (1 - self.w) * lr
        self.w += dw

    def t_stdp_forgetting(self, lr=.01, asymmetry=1):
        self.compute_slow_variable()
        dw = 0
        if self.presynaptic.get_spike_status():
            dw -= self.postsynaptic.get_output_current() * self.w * lr * asymmetry
        if self.postsynaptic.get_spike_status():
            dw += self.presynaptic.get_output_current() * self.slow_var * (1 - self.w) * lr
        self.forgetting()
        self.w += dw
        


    '''
    BCM rule related functions:
    '''
    def bcm(self):
        return 0

    '''
    Get info:
    '''
    def get_weight(self):
        return self.w
    
    def get_slow_variable(self):
        return self.slow_var
    
    def get_ids(self):
        return self.presynaptic.get_id(), self.postsynaptic.get_id()
    
    '''
    Manual changes:
    '''
    def set_weight_manually(self, weight):
        self.w = weight