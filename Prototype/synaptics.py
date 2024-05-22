import numpy as np

class Synapse:
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.scale = kwargs.get('scale', 1)
        self.slow_var = 0
        self.slow_tau = kwargs.get('slow_tau', 100)
        self.forget_tau = kwargs.get('forget_tau', 10000)
        #self.w = np.random.uniform(0.4, .6)
        self.w = 1
        self.slow_variable_limit = kwargs.get('slow_variable_limit', False)
        if self.slow_variable_limit:
            self.slow_var_choice = self.slow_var_limited
        else:
            self.slow_var_choice = self.slow_var_unlimited
        self.lr = .01
        self.learning_rule = None
        self.learning_rules = {None: self.empty_fun,
                               'pair_stdp': self.pair_stdp,
                               't_stdp': self.t_stdp,
                               't_stdp_forget': self.t_stdp_forgetting,
                               'bcm': self.bcm}


    def forward(self):
        self.postsynaptic.accumulate_current(self.presynaptic.get_output_current() * self.w * self.scale)
        self.learning_rules[self.learning_rule]()

    def forgetting(self):
        dw = -(self.w * self.postsynaptic.get_output_current()) / self.forget_tau
        self.w += dw

    '''
    Pair-based STDP rule:
    '''
    def pair_stdp(self, asymmetry=5):
        dw = 0
        if self.presynaptic.get_spike_status():
            dw -= self.postsynaptic.get_output_current() * self.w * self.lr * asymmetry
        if self.postsynaptic.get_spike_status():
            dw += self.presynaptic.get_output_current() * (1 - self.w) * self.lr
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

    def t_stdp(self, asymmetry=1):
        self.compute_slow_variable()
        dw = 0
        if self.presynaptic.get_spike_status():
            dw -= self.postsynaptic.get_output_current() * self.w * self.lr * asymmetry
        if self.postsynaptic.get_spike_status():
            dw += self.presynaptic.get_output_current() * self.slow_var * (1 - self.w) * self.lr
        self.w += dw

    def t_stdp_forgetting(self, asymmetry=1):
        self.compute_slow_variable()
        dw = 0
        if self.presynaptic.get_spike_status():
            dw -= self.postsynaptic.get_output_current() * self.w * self.lr * asymmetry
        if self.postsynaptic.get_spike_status():
            dw += self.presynaptic.get_output_current() * self.slow_var * (1 - self.w) * self.lr
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

    def change_lr(self, lr):
        self.lr = lr

    def change_learning_rule(self, rule):
        self.learning_rule = rule

    '''
    Misc:
    '''
    def empty_fun(self):
        pass




class Delayed_synapse(Synapse):
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        super().__init__(presynaptic, postsynaptic, **kwargs)
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.delay = kwargs.get('delay', 250)
        self.max_delay = kwargs.get('max_delay', 10)
        self.pre_impulse_queue = [0 for i in range(self.max_delay)]
        self.pre_spiked = [0 for i in range(self.max_delay)]
        self.max_delay -= 1

        # Debug:
        self.dd = 0
        self.delay_debug = 0

    def forward(self):
        self.pre_impulse_queue.pop(0)
        self.pre_spiked.pop(0)
        self.pre_impulse_queue.append(self.presynaptic.get_output_current())
        self.pre_spiked.append(self.postsynaptic.get_spike_status())
        self.postsynaptic.accumulate_current(self.pre_impulse_queue[int(self.max_delay - self.delay)] * self.w * self.scale)
        self.sophisticated_rule()
        self.learning_rules[self.learning_rule]()

    def sophisticated_rule(self, lr=1, asymmetry=1, alpha = 1):
        delay = self.delay / self.max_delay
        self.delay_debug = delay
        dd = 0
        if self.pre_spiked[int(self.max_delay - self.delay)]:
            #print(self.postsynaptic.get_output_current())
            dd -= (1 - self.postsynaptic.get_output_current()) * delay * lr * asymmetry
            self.dd = dd
            #print(dd)
            
        if self.postsynaptic.get_spike_status():
        #    print(self.pre_impulse_queue[int(self.max_delay - self.delay)])
            dd += (1 - self.pre_impulse_queue[int(self.max_delay - self.delay)]) * (1 - delay) * lr * alpha
            self.dd = dd
            
            
        self.delay += dd

'''
    def sophisticated_rule(self, lr=.1, asymmetry=50):
        delay = self.delay / self.max_delay
        self.delay_debug = delay
        dd = 0
        if self.presynaptic.get_spike_status():
            dd -= (1 - self.postsynaptic.get_output_current()) * delay * lr * asymmetry
            self.dd = dd
        if self.postsynaptic.get_spike_status():
            dd += (1 - self.presynaptic.get_output_current()) * (1 - delay) * lr
            self.dd = dd
        self.delay += dd


    def sophisticated_rule(self, lr=.1, asymmetry=50):
        delay = self.delay / self.max_delay
        self.delay_debug = delay
        dd = 0
        if self.presynaptic.get_spike_status():
            dd -= (1 - self.postsynaptic.get_output_current()) * delay * lr * asymmetry
            self.dd = dd
        if self.postsynaptic.get_spike_status():
            dd += (1 - self.pre_impulse_queue[int(self.max_delay - self.delay)]) * (1 - delay) * lr
            self.dd = dd
        self.delay += dd
'''