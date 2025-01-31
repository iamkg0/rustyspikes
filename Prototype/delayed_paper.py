from model import *
from vis_functions import *

res = .1


'''
MODEL AND PROTOCOL FOR RULE DEMONSTRATION
Includes single synapse of two pseudoneurons
'''
def exp11(rt=50, aw_in=0, aw_out=6, tau=10, synaptic_limit=1, scale=1, delay=10, max_delay=100, b=0):
    # initializes the model
    snn = SNNModel()
    output = Spikes_at_will(awaiting_time=aw_out, refresh_time=rt, tau=tau, synaptic_limit=synaptic_limit)
    input = Spikes_at_will(awaiting_time=aw_in, refresh_time=rt, tau=tau, synaptic_limit=synaptic_limit)
    synapse = Delayed_synapse(input, output, scale=scale, delay=delay, max_delay=max_delay, b=b)
    snn.add_neuron(output)
    snn.add_neuron(input)
    snn.add_synapse(synapse)
    snn.reload_graph()
    return snn

def protocol_11(model, time=100, plot=True, return_gatherer=False):
    delay = []
    dd = []
    gatherer = Gatherer(model)
    t = np.arange(int(time/res)) * res
    for i in t:
        model.tick()
        if plot:
            gatherer.gather_stats()
        delay.append(model.syn_by_edge[1,0].delay)
        dd.append(model.syn_by_edge[1,0].dd)
    if plot:
        draw_stats_gatherer(*gatherer.get_stats(pre_ids=[1], post_ids=[0]), t)
    if return_gatherer:
        return model, delay, np.array(dd), gatherer
    else:
        return model, delay, np.array(dd)


'''
MODEL AND PROTOCOL FOR DEMONSTATION OF b VALUE
Includes Izhikevich neuron on postsynapse
'''
def exp12(sc=7, rt=100, aw=1, tau=10, synaptic_limit=1, delay=20, max_delay=100, d_lr=1, b=6.7):
    # OPTIMAL PARAMS FOR 0 IMPACT:
    # rt0 = 100 msec, aw = 1 msec, tau = 10, delay = 0 (any, in fact), b = 6.7
    snn = SNNModel()
    output = Izhikevich(tau=tau, synaptic_limit=synaptic_limit)
    input = Spikes_at_will(awaiting_time=aw, refresh_time=rt, synaptic_limit=synaptic_limit, tau=tau)
    syn = Delayed_synapse(input, output, scale=sc, delay=delay, max_delay=max_delay, d_lr=d_lr, b=b)
    snn.add_neuron(output)
    snn.add_neuron(input)
    snn.add_synapse(syn)
    snn.reload_graph()
    return snn

def protocol_12(model, time=100, plot=True, return_gatherer=False):
    delay = []
    dd = []
    gatherer = Gatherer(model)
    t = np.arange(int(time/res)) * res
    for i in t:
        model.tick()
        if plot:
            gatherer.gather_stats()
        delay.append(model.syn_by_edge[1,0].delay)
        dd.append(model.syn_by_edge[1,0].dd)
    if plot:
        draw_stats_gatherer(*gatherer.get_stats(pre_ids=[1], post_ids=[0]), t)
    if return_gatherer:
        return model, delay, np.array(dd), gatherer
    else:
        return model, delay, np.array(dd)
    

'''
Sample random inputs
Stores in patterns (dict)
'''
def sampler(num_inputs=10, num_rand_patterns=2):
    num_patterns = num_rand_patterns + 1
    assert num_rand_patterns < np.math.factorial(num_inputs) - 1, 'too many patterns for an input of this size'
    sequence = np.arange(num_inputs) + 1
    patterns = {}
    patterns[0] = sequence.tolist()
    for p in range(1, num_rand_patterns+2):
        seq = sequence.copy()
        np.random.shuffle(seq)
        seq = seq.tolist()
        if seq not in patterns.values():
            patterns[p] = seq
        else:
            while seq in patterns.values():
                np.random.shuffle(seq)
                if seq not in patterns.values():
                    patterns[p] = seq
                    break
    return patterns, num_inputs, num_patterns


'''
MODEL FOR DEMONSTRATION OF ROBUSTNESS OF THE DELAYED-BASED LEARNING RULE
'''
def exp21(sc=7, num_inputs=10, tau=10, synaptic_limit=1, delay=1, max_delay=100, d_lr=.1, b=4.6):
    snn = SNNModel()
    output = Izhikevich(tau=tau, synaptic_limit=synaptic_limit)
    snn.add_neuron(output)
    for saw in range(num_inputs):
        neu = Spikes_at_will(tau=tau, synaptic_limit=synaptic_limit)
        snn.add_neuron(neu)
        syn = Delayed_synapse(neu, output, scale=sc, delay=delay, max_delay=max_delay, d_lr=d_lr, b=b)
        snn.add_synapse(syn)
    snn.reload_graph()
    return snn


'''
MODEL WITH PAIR-BASED OR TRIPLET STDP RULE
Notice that the exact rule is set in protocol, which works for any rule in the experiment
'''
def exp22(sc=7, num_inputs=10, tau=10, synaptic_limit=1):
    snn = SNNModel()
    output = Izhikevich(tau=tau, synaptic_limit=synaptic_limit)
    snn.add_neuron(output)
    for weighted in range(num_inputs):
        neu = Spikes_at_will()
        snn.add_neuron(neu)
        syn = Synapse(neu, output, scale=sc)
        snn.add_synapse(syn)
    snn.reload_graph()
    return snn


'''
THE PROTOCOL
'''
def run_protocol(model, sampler, num_patterns=3, sample_time=150, interval=6, runs=3, lr=.1, d_lr=None,
                 freeze_delays=False, gather_data=False, plot=False, plast_type=None, return_gatherer=False, gather_delays=True):
    model.set_rule_to_all(plast_type)
    model.set_lr_to_all(lr)
    if d_lr:
        model.set_d_lr(d_lr)
    delay = [[] for i in range(len(model.show_config()['Neurons'])-1)]
    dd = [[] for i in range(len(model.show_config()['Neurons'])-1)]
    num_spikes = [0 for i in range(len(sampler))]

    if gather_data:
        gatherer = Gatherer(model)
    for run in range(runs):
        for p in range(len(sampler)):
            sample = sampler[p]
            aw = 1
            for neu in sample:
                model.neurons[neu].awaiting_time = aw
                model.neurons[neu].refresh()
                aw += interval
            for t in np.arange(int(sample_time/res)) * res:
                model.tick(freeze_delays=freeze_delays)
                if model.neurons[0].get_spike_status():
                    num_spikes[p] += 1
                if gather_data:
                    gatherer.gather_stats()
                if not freeze_delays:
                    if gather_delays:
                        for edge in range(1, len(model.show_config()['Neurons'])):
                            #print(np.array(delay).shape, np.array(dd).shape)
                            delay[edge-1].append(model.syn_by_edge[edge,0].delay)
                            dd[edge-1].append(model.syn_by_edge[edge,0].dd)
    if plot and gather_data:
        draw_stats_gatherer(*gatherer.get_stats(pre_ids=list(range(1, len(model.show_config()['Neurons']))),
                                                post_ids=[0]), time_range=sample_time*runs*(len(sampler)), resolution=res)
    if return_gatherer and gather_data:
        return model, np.array(delay), np.array(dd), np.array(num_spikes), gatherer
    else:
        return model, np.array(delay), np.array(dd), np.array(num_spikes)