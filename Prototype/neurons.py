import random
import numpy as np


class Neuron:
    def __init__(self, **kwargs):
        self.resolution = kwargs.get('resolution', .1)
        self.noise = kwargs.get('noise', 0)
        self.tau = kwargs.get('tau', 30)
        self.syn_out = kwargs.get('syn_out', True)
        self.I = kwargs.get('I', 0)
        self.spiked = False
        self.impulse = 0
        self.synaptic_limit = kwargs.get('synaptic_limit', None)
        if self.synaptic_limit:
            self.spike_trace_choice = self.spike_trace_limited
        else:
            self.spike_trace_choice = self.spike_trace_unlimited
        self.cum_current = 0
        self.id = kwargs.get('id', None)
        self.preset = kwargs.get('preset', None)

    '''
    Synaptic output related functions:
    '''
    def spike_decrease(self):
        self.impulse -= (self.impulse / self.tau) * self.resolution
        self.spiked = False
        return self.impulse
    
    def spike_trace_limited(self):
        self.impulse = self.synaptic_limit
        self.spiked = True
    
    def spike_trace_unlimited(self):
        self.impulse += 1
        self.spiked = True
    
    '''
    Update input current:
    '''
    def apply_current(self, current):
        self.I = current

    def accumulate_current(self, current):
        self.cum_current += current

    def apply_cum_current(self):
        self.I = self.cum_current
        self.cum_current = 0

    '''
    Get info:
    '''
    def get_output_current(self):
        '''
        Returns current value
        '''
        return self.impulse
    
    def get_spike_status(self):
        '''
        Returns the status of neuron - if it has spiked at this iteration or not
        '''
        return self.spiked
    
    def get_voltage_dynamics(self):
        '''
        Returns voltage of a neuron
        '''
        return self.v
    
    def get_input_current(self):
        '''
        Returns the amount of current this neuron recieves
        '''
        return self.I
    
    def get_id(self):
        return self.id


class Izhikevich(Neuron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        preset = kwargs.get('preset', None)
        self.ap_threshold = kwargs.get('ap_threshold', 30)
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
        self.v = -70
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


class LIF(Neuron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lif_tau = kwargs.get('tau', 30)
        self.v = 0
        self.v_eq = 0
        self.ap_threshold = kwargs.get('ap_threshold', 1)

    def dynamics(self):
        self.v -= (self.v - self.v_eq)/self.lif_tau + self.I
        if self.v >= self.ap_threshold:
            self.v = self.v_eq
            self.spike_trace_choice()
        else:
            self.spike_decrease()



class Probability_neuron(Neuron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v = 0


    def dynamics(self):
        self.v = random.choices([0, 1], [1-self.I, self.I])[0]
        if self.v == 1:
            self.spike_trace_choice()
        else:
            self.spike_decrease()
        return self.impulse
    


class Spikes_at_will(Neuron):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.awaiting_time = kwargs.get('awaiting_time', 10)
        self.awaiting_timer = self.awaiting_time
        self.awaiting_time_passed = False
        self.refresh_time = kwargs.get('refresh_time', 300)
        self.refresh_cooldown = self.refresh_time
        self.spike_occured = False
        self.v = 0
    
    def dynamics(self):
        self.refresh_cooldown -= self.resolution
        self.v = 0
        if not self.spike_occured:
            self.awaiting_timer -= self.resolution
            if self.awaiting_timer <= 0:
                self.spike_trace_choice()
                self.v = 1
                self.spike_occured = True
            else:
                self.spike_decrease()
        else:
            self.spike_decrease()
        if self.refresh_cooldown <= 0:
            self.refresh()
        return self.impulse
    
    def refresh(self):
        self.awaiting_timer = self.awaiting_time
        self.refresh_cooldown = self.refresh_time
        self.spike_occured = False

    def change_props(self, **kwargs):
        self.awaiting_time = kwargs.get('awaiting_time', self.awaiting_time)
        self.refresh_time = kwargs.get('refresh_time', self.refresh_time)
        self.refresh()