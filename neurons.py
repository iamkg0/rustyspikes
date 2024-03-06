import numpy as np

class Izhikevich:
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
        self.spiked = 0
        self.impulse = 0
        self.I = kwargs.get('I', 0)

    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I) + np.random.uniform(-self.noise, self.noise, size=1)[0]
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        if self.v >= self.ap_threshold:
            self.v = self.c
            self.u += self.d
            return 1
        else:
            return 0
        
    def apply_current(self, current):
        self.I = current

    def propagate(self):
        return self.transmitter_impact
    
    def spike_trace(self):
        self.impulse -= 0