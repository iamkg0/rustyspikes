import random

class Izhikevich:
    def __init__(self, **kwargs):
        self.resolution = kwargs.get('resolution', .1)
        self.noise = kwargs.get('noise', 0)
        self.ap_threshold = kwargs.get('ap_threshold', 30)
        self.tau = kwargs.get('tau', 30)
        self.syn_out = kwargs.get('syn_out', True)
        self.synaptic_limit = kwargs.get('synaptic_limit', None)
        self.I = kwargs.get('I', 0)
        self.spiked = 0
        self.impulse = 0
        if self.synaptic_limit:
            self.spike_trace_choice = self.spike_trace_limited
        else:
            self.spike_trace_choice = self.spike_trace_unlimited
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        preset = kwargs.get('preset', None)
        param_list = [[0.02, 0.2, -65, 8],
						[0.02, 0.2, -55, 4],
						[0.02, 0.2, -50, 2],
						[0.1, 0.2, -65, 2],
						[0.02, 0.25, -65, 0.05],
						[0.1, 0.26, -65, 8],
						[0.02, 0.25, -65, 2],
						[kwargs.get('a', .02), kwargs.get('b', .2), kwargs.get('c', -65), kwargs.get('d', 2)]]
        idx = preset_list.index(preset)
        assert preset in preset_list, f'Preset {preset} does not exist! Use one from {preset_list}'
        self.a = param_list[idx][0]
        self.b = param_list[idx][1]
        self.c = param_list[idx][2]
        self.d = param_list[idx][3]
        self.v = self.c
        self.u = self.b * self.v
        
        

    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I) + random.uniform(-self.noise, self.noise)
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        if self.v >= self.ap_threshold:
            self.v = self.c
            self.u += self.d
            self.spike_trace_choice()
        else:
            self.spike_decrease()
        return self.impulse
        
    def apply_current(self, current):
        self.I = current
    
    '''
    Synaptic output related functions:
    '''
    def spike_decrease(self):
        self.impulse -= self.impulse / self.tau
        return self.impulse
    
    def spike_trace_limited(self):
        self.impulse = self.synaptic_limit
    
    def spike_trace_unlimited(self):
        self.impulse += 1





class Poisson_neuron:
    def __init__(self, **kwargs):
        self.resolution = kwargs.get('resolution', .1)
        self.excitatory = kwargs.get('excitatory', True)
        if self.excitatory:
            self.transmitter_impact = 1
        else:
            self.transmitter_impact = -1
        self.noise = kwargs.get('noise', 0)
        self.ap_threshold = kwargs.get('ap_threshold', 30)
        self.tau = kwargs.get('tau', 30)
        self.syn_out = kwargs.get('syn_out', True)
        self.I = kwargs.get('I', 0)