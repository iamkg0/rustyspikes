import numpy as np
import matplotlib.pyplot as plt
from utils import *
res = .1

''' SHORTCUTS '''
def short_single_synapse(pre_neu, post_neu, synapse, time, res=.1, rule=None, fwidth=16, fheight=9):
    '''
    Simulates a single synapse between two neurons and shows statistics
    '''
    vs, Is, spikes, w, t = simulate_synapse(pre_neu, post_neu, synapse, time, res, rule)
    show_stats_synapse(vs, Is, spikes, w, t, fwidth, fheight)


class Gatherer:
    def __init__(self, model):
        self.model = model
        self.vs = [[] for i in self.model.show_config()['Neurons']]
        self.impulses = [[] for i in self.model.show_config()['Neurons']]
        self.Is = [[] for i in self.model.show_config()['Neurons']]
        self.weights = [[] for i in self.model.show_config()['Synapses']]
        self.spikes = [[] for i in self.model.show_config()['Neurons']]
        self.weight_index_by_edge = {}
        counter = 0
        for i in self.model.show_config()['Synapses']:
            self.weight_index_by_edge[i] = counter
            counter += 1
        self.timer = 0

    def gather_stats(self, gather_delay=False):
        for i in self.model.show_config()['Neurons']:
            self.vs[i].append(self.model.neurons[i].get_voltage_dynamics())
            self.impulses[i].append(self.model.neurons[i].get_output_current())
            self.Is[i].append(self.model.neurons[i].get_input_current())
            self.spikes[i].append(self.model.neurons[i].get_spike_status())
        counter = 0
        if not gather_delay:
            for w in self.model.show_config()["Synapses"]:
                self.weights[counter].append(self.model.syn_by_edge[w].get_weight())
                counter += 1
        if gather_delay:
            for w in self.model.show_config()["Synapses"]:
                self.weights[counter].append(self.model.syn_by_edge[w].get_delay)
        self.timer += res

    def get_stats(self, pre_ids = None, post_ids = None, weight_ids = None, timings=None):
        if not timings:
            timings = list(range(len(self.vs[0])))
        if not pre_ids:
            pre_ids = list(range(len(self.vs)-1))
        if not post_ids:
            post_ids = [len(pre_ids)]
        if not weight_ids:
            weight_ids = [i for i in self.weight_index_by_edge] #list of tuples with ids
        weight_indexes = []
        for i in weight_ids:
            weight_indexes.append(self.weight_index_by_edge[i])
        pre_vs = np.array(self.vs)[pre_ids]
        pre_impulses = np.array(self.impulses)[pre_ids]
        pre_Is = np.array(self.Is)[pre_ids]
        post_vs = np.array(self.vs)[post_ids]
        post_impulses = np.array(self.impulses)[post_ids]
        post_Is = np.array(self.Is)[post_ids]
        ws = np.array(self.weights)[weight_indexes]
        return pre_vs[:,timings], pre_impulses[:,timings], pre_Is[:,timings], post_vs[:,timings], post_impulses[:,timings], post_Is[:,timings], ws[:,timings]
    
    def show_spike_stats(self, print_results=True):
        num_spikes = np.sum(np.array(self.spikes[-1]))
        step = self.timer / 1000
        freq = round(num_spikes / step, 3)
        if print_results:
            print(f'Total number of spikes = {num_spikes}, avg frequency = {freq}')
        return num_spikes, freq
    
    def drop_timer_and_spikes(self):
        self.timer = 0
        self.spikes = [[] for i in self.model.show_config()['Neurons']]

    def show_stats(self, timings=None):
        '''
        doesnt work ahahahaha
        '''
        if not timings:
            timings = range(len(self.pre_vs)-1)
        return self.pre_vs[timings], self.pre_impulses[timings], self.pre_Is[timings], self.post_vs[timings], self.post_impulses[timings], self.post_Is[timings], self.ws[timings]

    
def draw_convergence(data, total=True):
    if type(data) != np.ndarray:
        data = np.array(data)
    plus = [[] for i in range(data.shape[0])]
    minus = [[] for i in range(data.shape[0])]
    for i in range(data.shape[0]):
        temp = data[i, 0]
        prev_minus = data[i, 0]
        prev_plus = data[i, 0]
        for j in range(data.shape[1]):
            tick = temp + data[i, j]
            if temp < data[i, j]:
                minus[i].append(-tick)
                prev_minus = -tick
                plus[i].append(prev_plus)
            if temp > data[i, j]:
                plus[i].append(tick)
                prev_plus = tick
                minus[i].append(prev_minus)
            if temp == data[i, j]:
                plus[i].append(prev_plus)
                minus[i].append(prev_minus)
            temp = data[i, j]
    plus = np.array(plus)
    minus = np.array(minus)
    if total:
        plus = np.sum(plus, axis=0)
        minus = np.sum(minus, axis=0)
        plus_n = plus / np.max(plus)
        minus_n = minus / np.max(minus)
    else:
        plus_n = plus / np.max(plus, axis=1)
        minus_n = minus / np.max(minus, axis=1)
    
    plt.figure()
    plt.plot(plus_n)
    plt.plot(minus_n)
    plt.show()


def draw_stats_gatherer(pre_vs, pre_impulses, pre_Is, post_vs, post_impulses, post_Is, ws, time_range, resolution=None, fwidth=16, fheight=12, dpi=100):
    if not isinstance(time_range, np.ndarray) or isinstance(time_range, list):
        if not resolution:
            raise Exception('I require integration step or time array')
        time_range = np.arange(int(time_range / resolution)) * resolution
    figure, axis = plt.subplots(7, 1)
    plt.subplots_adjust(hspace=.5)
    figure.set_figwidth(fwidth)
    figure.set_figheight(fheight)
    figure.set_dpi(dpi)
    for i in range(pre_vs.shape[0]):
        pre_voltage = axis[0]
        pre_imps = axis[1]
        pre_inp_is = axis[2]
        pre_voltage.plot(time_range, pre_vs[i])
        #pre_imps.set_ylim(0, 3)
        pre_imps.plot(time_range, pre_impulses[i])
        pre_inp_is.plot(time_range, pre_Is[i])
    for j in range(post_vs.shape[0]):
        post_voltage = axis[3]
        post_imps = axis[4]
        post_inp_is = axis[5]
        post_voltage.plot(time_range, post_vs[j])
        #post_imps.set_ylim(0, 3)
        post_imps.plot(time_range, post_impulses[j])    
        #post_inp_is.set_ylim(0,3)
        post_inp_is.plot(time_range, post_Is[j])
    for w in range(ws.shape[0]):
        weights = axis[6]
        weights.plot(time_range, ws[w])
    pre_voltage.set_title('Presynaptic potential')
    pre_imps.set_title('Presynaptic output')
    pre_inp_is.set_title('Presynaptic input current')
    post_voltage.set_title('Postsynaptic potential')
    post_imps.set_title('Postsynaptic output')
    post_inp_is.set_title('Postsynaptic input current')
    weights.set_title('Weight changes')
    plt.show()

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


def barplot_annotate_brackets(num1, num2, data, center, height=None, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    # Stolen and adapted from somewhere on stackoverflow, I guess
    # Unfortunately, I can't find the tred
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1]+1, height[num1]
    rx, ry = center[num2]+1, height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly.all(), ry.all()) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)