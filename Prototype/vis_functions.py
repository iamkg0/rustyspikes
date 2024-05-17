import numpy as np
import matplotlib.pyplot as plt
from utils import *


''' SHORTCUTS '''
def short_single_synapse(pre_neu, post_neu, synapse, time, res=.1, rule=None, fwidth=16, fheight=9):
    '''
    Simulates a single synapse between two neurons and shows statistics
    '''
    vs, Is, spikes, w, t = simulate_synapse(pre_neu, post_neu, synapse, time, res, rule)
    show_stats_synapse(vs, Is, spikes, w, t, fwidth, fheight)


class Gatherer:
    def __init__(self, model, pre_ids, post_ids):
        '''This is stupid. I wil rewrite this'''
        self.model = model
        self.neurons = model.show_config()['Neurons']
        self.synapses = model.show_config()['Synapses']
        self.post_ids = post_ids
        self.pre_ids = pre_ids
        self.pre_vs = [[] for i in range(len(pre_ids))]
        self.post_vs = [[] for i in range(len(post_ids))]
        self.pre_impulses = [[] for i in range(len(pre_ids))]
        self.post_impulses = [[] for i in range(len(post_ids))]
        self.weights = [[] for i in range(len(self.synapses))]
        self.pre_is = [[] for i in range(len(pre_ids))]
        self.post_is = [[] for i in range(len(post_ids))]

    def gather_stats(self, pre_ids=None, post_ids=None):
        if not pre_ids:
            pre_ids = self.pre_ids
        if not post_ids:
            post_ids = self.post_ids
        counter = 0
        for pre in pre_ids:
            v, impulse, I = self.model.get_neuron_stats(pre)
            self.pre_vs[counter].append(v)
            self.pre_impulses[counter].append(impulse)
            self.pre_is[counter].append(I)
            counter += 1
        counter = 0
        for post in post_ids:
            v, impulse, I = self.model.get_neuron_stats(post)
            self.post_vs[counter].append(v)
            self.post_impulses[counter].append(impulse)
            self.post_is[counter].append(I)
            counter += 1

    def get_stats(self, pre_ids=None, post_ids=None):
        if not pre_ids:
            pre_ids = self.pre_ids
        if not post_ids:
            post_ids = self.post_ids
        pre_ids = self.neurons[*pre_ids]
        pre_vs = np.array(self.pre_vs)[pre_ids]
        pre_impulses = np.array(self.pre_impulses)[pre_ids]
        pre_is = np.array(self.pre_is)[pre_ids]
        print(pre_vs)
        post_vs = np.array(self.post_vs)[post_ids]
        post_impulses = np.array(self.post_impulses)[post_ids]
        post_is = np.array(self.post_is)[post_ids]
        return pre_vs, pre_impulses, pre_is, post_vs, post_impulses, post_is
            



'''
Single synapse simulation:
'''
def simulate_synapse(pre_neu, post_neu, synapse, time, res=.1, rule=None):
    rules = {None: None,
             'pair_stdp': synapse.pair_stdp,
             't_stdp': synapse.t_stdp,
             't_stdp_forgetting': synapse.t_stdp_forgetting,
             'bcm': synapse.bcm}
    use_rule = rules[rule]
    t = np.arange(int(time / res)) * res
    pre_traces = []
    post_traces = []
    pre_Is = []
    post_Is = []
    pre_v = []
    post_v = []
    w = []
    slow_traces = []
    for i in range(int(time / res)):
        pre_traces.append(pre_neu.dynamics())
        post_traces.append(post_neu.dynamics())
        pre_Is.append(pre_neu.get_input_current())
        post_Is.append(post_neu.get_input_current())
        pre_v.append(pre_neu.get_voltage_dynamics())
        post_v.append(post_neu.get_voltage_dynamics())
        synapse.forward()
        post_neu.apply_cum_current()
        if use_rule:
            use_rule()
            if rule == 't_stdp' or rule == 't_stdp_forgetting':
                slow_traces.append(synapse.get_slow_variable())
        w.append(synapse.get_weight())
    if rule == 't_stdp' or rule == 't_stdp_forgetting':
        traces_bundle = np.array(pre_traces), np.array(post_traces), np.array(slow_traces)
    else:
        traces_bundle = np.array(pre_traces), np.array(post_traces)
    traces = np.stack((traces_bundle), axis=0)
    Is = np.stack((pre_Is, post_Is), axis=0)
    vs = np.stack((pre_v, post_v), axis=0)
    w = np.array(w)
    return vs, Is, traces, w, t



def show_stats_synapse(vs, Is, spikes, w, t, fwidth=16, fheight=9):
    figure, axis = plt.subplots(4, 1)
    figure.set_figwidth(fwidth)
    figure.set_figheight(fheight)
    colors = ['cyan', 'orange', 'red']
    labels = ['pre', 'post', 'post_slow']
    for i in range(len(vs)):
        voltage = axis[0]
        cur = axis[1]
        spks = axis[2]
        voltage.plot(t, vs[i], color=colors[i], label=labels[i], linewidth=1)
        cur.plot(t, Is[i], color=colors[i], label=labels[i], linewidth=1)
        spks.plot(t, spikes[i], color=colors[i], label=labels[i], linewidth=1)
    if len(vs) < len(spikes):
        spks.plot(t, spikes[i+1], color=colors[i+1], label=labels[i+1], linewidth=1)
    voltage.legend(loc='upper right')
    cur.legend(loc='upper right')
    spks.legend(loc='upper right')
    w_ax = axis[3]
    w_ax.plot(t, w, color=colors[0], linewidth=1)
    voltage.set_title('Voltage')
    cur.set_title('Input current')
    spks.set_title('Spikes')
    w_ax.set_title('Weight changes')


'''
Forgot what this is:
'''
def dev_test(neuron, time, timings, currents, default_current=0):
    t = np.arange(int(time / neuron.resolution)) * neuron.resolution
    vs = []
    spikes = []
    Is = []
    var_idx = 0
    for step in range(len(t)):
        #neuron.apply_current(default_current)
        if len(timings) > var_idx:
            if step == int(timings[var_idx] / neuron.resolution):
                neuron.apply_current(currents[var_idx])
                var_idx += 1
        spikes.append(neuron.dynamics())
        Is.append(neuron.I)
        vs.append(neuron.v)
        
    return vs, Is, spikes,t 


def show_stats(vs, Is, spikes, t, fwidth=15, fheight=9):
    figure, axis = plt.subplots(3, 1)
    figure.set_figwidth(fwidth)
    figure.set_figheight(fheight)
    voltage = axis[0]
    cur = axis[1]
    spks = axis[2]
    voltage.plot(t, vs)
    voltage.set_title('Voltage')
    cur.plot(t, Is)
    cur.set_title('Input current')
    spks.plot(t, spikes)
    spks.set_title('Spikes')



def dev_two(pre, post, time, ap_scale=1):
    t = np.arange(int(time / pre.resolution)) * pre.resolution
    vs_pre = []
    spikes_pre = []
    Is_pre = []
    vs_post = []
    spikes_post = []
    Is_post = []
    var_idx = 0
    for step in range(len(t)):
        #neuron.apply_current(default_current)
        ap = pre.dynamics()
        spikes_pre.append(ap)
        Is_pre.append(pre.I)
        vs_pre.append(pre.v)
        post.apply_current(ap * ap_scale)
        spikes_post.append(post.dynamics())
        Is_post.append(post.I)
        vs_post.append(post.v)
        
    return vs_pre, Is_pre, spikes_pre, vs_post, Is_post, spikes_post, t



def simulate_single(neu, time, ap_scale=1):
    t = np.arange(int(time / neu.resolution)) * neu.resolution
    vs = []
    spikes = []
    Is = []
    for step in range(len(t)):
        ap = neu.dynamics()
        spikes.append(ap * ap_scale)
        Is.append(neu.I)
        vs.append(neu.v)
        
    return vs, Is, spikes, t



def simulate_several(pre, post, time, ap_scale=[1]):
    t = np.arange(int(time / post.resolution)) * post.resolution
    vs_pre = []
    spikes_pre = []
    Is_pre = []
    vs_post = []
    spikes_post = []
    Is_post = []
    var_idx = 0
    for neus in range(len(pre)):
        vs = []
        spikes = []
        Is = []
        for step in range(len(t)):
            #neuron.apply_current(default_current)
            ap = pre[neus].dynamics()
            spikes.append(ap)
            Is.append(pre[neus].I)
            vs.append(pre[neus].v)
        spikes_pre.append(spikes)
        Is_pre.append(Is)
        vs_pre.append(vs)

    aps = np.array(spikes_pre)
    aps = aps.sum(0)
    print(aps.shape)
    for step in range(len(t)):
        post.apply_current(aps[step] * ap_scale)
        spikes_post.append(post.dynamics())
        Is_post.append(post.I)
        vs_post.append(post.v)
        
    return vs_pre, Is_pre, spikes_pre, vs_post, Is_post, spikes_post, t



def show_stats_many(vs, Is, spikes, t, fwidth=15, fheight=9):
    figure, axis = plt.subplots(3, 1)
    figure.set_figwidth(fwidth)
    figure.set_figheight(fheight)
    for i in range(len(vs)):
        voltage = axis[0]
        cur = axis[1]
        spks = axis[2]
        voltage.plot(t, vs[i])
        cur.plot(t, Is[i])
        spks.plot(t, spikes[i])
    voltage.set_title('Voltage')
    cur.set_title('Input current')
    spks.set_title('Spikes')