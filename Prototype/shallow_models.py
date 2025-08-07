from model import *

def net_v0():
    '''
    5-to-1 test subject
    '''
    snn = SNNModel()
    rt=100
    scale = 10
    in0 = Spikes_at_will(awaiting_time=2, refresh_time=rt)
    in1 = Spikes_at_will(awaiting_time=6, refresh_time=rt)
    in2 = Spikes_at_will(awaiting_time=10, refresh_time=rt)
    in3 = Spikes_at_will(awaiting_time=14, refresh_time=rt)
    in4 = Spikes_at_will(awaiting_time=16, refresh_time=rt)
    out = Izhikevich()
    syn0 = Synapse(in0, out, scale=scale)
    syn1 = Synapse(in1, out, scale=scale)
    syn2 = Synapse(in2, out, scale=scale)
    syn3 = Synapse(in3, out, scale=scale)
    syn4 = Synapse(in4, out, scale=scale)
    snn.add_neuron(in0)
    snn.add_neuron(in1)
    snn.add_neuron(in2)
    snn.add_neuron(in3)
    snn.add_neuron(in4)
    snn.add_neuron(out)
    snn.add_synapse(syn0)
    snn.add_synapse(syn1)
    snn.add_synapse(syn2)
    snn.add_synapse(syn3)
    snn.add_synapse(syn4)
    snn.reload_graph()
    return snn

def single_synapse():
    '''
    1-to-1 test subject
    '''
    snn = SNNModel()
    rt = 100
    input = Spikes_at_will(awaiting_time=5, refresh_time=rt, tau=30)
    output = Spikes_at_will(awaiting_time=2, refresh_time=rt, tau=30)
    syn = Synapse(input, output, scale=1)
    snn.add_neuron(input)
    snn.add_neuron(output)
    snn.add_synapse(syn)
    snn.reload_graph()
    return snn

def dr_single():
    snn = SNNModel()
    rt0 = 100
    aw0 = 1
    rt1 = 100
    aw1 = 10
    input = Spikes_at_will(awaiting_time=aw0, refresh_time=rt0, synaptic_limit=1, tau=10)
    output = Spikes_at_will(awaiting_time=aw1, refresh_time=rt1, synaptic_limit=1, tau=10)
    syn = Delayed_synapse(input, output, scale=1, delay=7.4, max_delay=200)
    snn.add_neuron(input)
    snn.add_neuron(output)
    snn.add_synapse(syn)
    snn.reload_graph()
    return snn

def dr_izh_single(sc=7, aw0=1, rt0=100, b=7.4, d_lr=1, delay=1):
    snn = SNNModel()
    #rt0 = 100 # msec
    #aw0 = 1 # msec
    input = Spikes_at_will(awaiting_time=aw0, refresh_time=rt0, synaptic_limit=1, tau=10)
    output = Izhikevich(tau=10, synaptic_limit=1)
    syn = Delayed_synapse(input, output, scale=sc, delay=delay, max_delay=100, d_lr=d_lr, b=b)
    snn.add_neuron(input)
    snn.add_neuron(output)
    snn.add_synapse(syn)
    snn.define_output(1)
    snn.reload_graph()
    return snn


def delayed_3to1():
    snn = SNNModel()
    rt = 100
    awaitings = [1, 20, 19, 30]
    delays = [19.0, 1.0, 21.0]
    neurons = []
    for i in awaitings:
        n = Spikes_at_will(awaiting_time=i, refresh_time=rt, synaptic_limit=1, tau=10)
        neurons.append(n)
        snn.add_neuron(n)
    for j in range(len(delays)):
        s = Delayed_synapse(neurons[j], neurons[-1], scale=1, delay=delays[j], max_delay=300)
        snn.add_synapse(s)
    snn.reload_graph()
    return snn

def delayed_10_inputs(rt=100, aw=10, dels=100, m_delay=300, scale=6, b=7.4, d_lr=10):
    snn = SNNModel()
    awaitings = np.arange(10) * aw
    delays = np.ones(10) * dels
    neurons = []
    for i in awaitings:
        n = Spikes_at_will(awaiting_time=i, refresh_time=rt, synaptic_limit=1, tau=10)
        neurons.append(n)
        snn.add_neuron(n)
    out = Izhikevich()
    snn.add_neuron(out)
    snn.define_output(out.id)
    for j in range(len(delays)):
        s = Delayed_synapse(neurons[j], out, scale=scale, b=b, refresh_time=rt, synaptic_limit=1, tau=10, delay=30, max_delay=m_delay, d_lr=d_lr)
        snn.add_synapse(s)
    snn.reload_graph()
    return snn



def delayed_single():
    '''
    1-to-1 delayed synapse test
    '''
    dsnn = SNNModel()
    rt = 100
    input = Spikes_at_will(awaiting_time=10, refresh_time=rt, synaptic_limit=1)
    #output = Spikes_at_will(awaiting_time=15, refresh_time=rt, synaptic_limit=1)
    output = Izhikevich()
    d_syn = Delayed_synapse(input, output, scale = 15, delay=5)
    dsnn.add_neuron(input)
    dsnn.add_neuron(output)
    dsnn.add_synapse(d_syn)
    dsnn.reload_graph()
    return dsnn

def delayed_3_to_1(out_type='SAW'):
    dsnn = SNNModel()
    rt = 100
    sc = 3
    in0 = Spikes_at_will(awaiting_time=2, refresh_time=rt, synaptic_limit=1)
    in1 = Spikes_at_will(awaiting_time=6, refresh_time=rt, synaptic_limit=1)
    in2 = Spikes_at_will(awaiting_time=10, refresh_time=rt, synaptic_limit=1)
    if out_type == 'IZH':
        out = Izhikevich(synaptic_limit=1)
    if out_type == 'SAW':
        out = Spikes_at_will(awaiting_time=13, refresh_time=rt, synaptic_limit=1)
    syn0 = Delayed_synapse(in0, out, scale=sc, delay=1)
    syn1 = Delayed_synapse(in1, out, scale=sc, delay=1)
    syn2 = Delayed_synapse(in2, out, scale=sc, delay=1)
    dsnn.add_neuron(in0)
    dsnn.add_neuron(in1)
    dsnn.add_neuron(in2)
    dsnn.add_neuron(out)
    dsnn.add_synapse(syn0)
    dsnn.add_synapse(syn1)
    dsnn.add_synapse(syn2)
    dsnn.reload_graph()
    return dsnn

def bfnaics_24_model(num_input, rt=100, scale=10):
    snn = SNNModel()
    neu = []
    for i in range(num_input):
        neu.append(Spikes_at_will(id=i, refresh_time=rt))
        snn.add_neuron(neu[i])
    out = Izhikevich()
    snn.add_neuron(out)
    for i in range(num_input):
        syn = Synapse(neu[i], out, scale=scale)
        snn.add_synapse(syn)
    snn.reload_graph()
    return snn



def lif_test(num_input=10, rt=100, scale=1, aw=5):
    snn = SNNModel()
    neu = []
    for i in range(num_input):
        neu.append(Spikes_at_will(id=i, refresh_time=rt, awaiting_time = i*aw))
        snn.add_neuron(neu[i])
    out = LIF(id=i+1)
    snn.add_neuron(out)
    for i in range(num_input):
        syn = Synapse(neu[i], out, scale=scale)
        snn.add_synapse(syn)
    snn.reload_graph()
    return snn



def conv_dyn(num_input=5, rt=100, scale=1, aw=5, pack_size=3, pack_step=2, num_hidden=5):
    snn = SNNModel()
    neu_in = []
    neu_h = []
    for i in range(num_input):
        neu_in.append(Spikes_at_will(id=i, refresh_time=rt, awaiting_time=aw*i))
        snn.add_neuron(neu_in[i])
    for j in range(num_hidden):
        neu_h.append(Izhikevich(id=num_input+j))
        snn.add_neuron(neu_h[j])
    out = Izhikevich(id=num_input+num_hidden)
    snn.add_neuron(out)
    current_pack = 0
    h=0
    while current_pack <= num_input - pack_size:
        for k in range(current_pack, pack_size+current_pack):
            syn = Synapse(neu_in[k], neu_h[h], scale=scale)
            snn.add_synapse(syn)
        h += 1
        current_pack += pack_step
    for m in range(num_hidden):
        syn = Synapse(neu_h[m], out, scale=scale)
        snn.add_synapse(syn)
    snn.define_output(ids=[num_input+num_hidden])
    snn.reload_graph()
    return snn


def one_neu_dynamics(num_input=6, scale=1.2, rt=100, interval=5, learning_rule='pair_stdp', delayed=False,
                     lr=.1, tau=30, d_lr=.1, synaptic_limit=False, slow_variable_limit=False, max_delay=100, b=7.6,
                     slow_tau=100, forget_tau=100, delay=1, noise=3, weights=(.4, .6), stick_del_w_to_one=True):
    snn = SNNModel()
    neu_in = []
    neu_out = Izhikevich(id=0, noise=noise)
    snn.add_neuron(neu_out)
    for i in range(num_input):
        neu_in.append(Spikes_at_will(id=i+1, refresh_time=rt, awaiting_time=interval * i))
        snn.add_neuron(neu_in[i])
    snn.define_output(ids=[0])
    if delayed:
        if stick_del_w_to_one:
            weight = 1
        else:
            weight = np.random.uniform(*weights)
        for i in range(num_input):
            syn = Delayed_synapse(neu_in[i], neu_out, scale=scale, synaptic_limit=1,
                                  delay=delay, max_delay=max_delay, b=b, tau=tau, d_lr=d_lr, w=weight)
            snn.add_synapse(syn)
    else:
        for i in range(num_input):
            syn = Synapse(neu_in[i], neu_out, scale=scale, synaptic_limit=synaptic_limit, slow_variable_limit=slow_variable_limit,
                          lr=lr, slow_tau=slow_tau, learning_rule=learning_rule, forget_tau=forget_tau, w=np.random.uniform(*weights))
            snn.add_synapse(syn)
    snn.reload_graph()
    return snn


def rf_test_unit(scale=1, rt=100, interval=5,
                 lr=.1, tau=30, d_lr=1, synaptic_limit=1, max_delay=100, b=7.6,
                 slow_tau=100, forget_tau=100, delay=0, noise=0, weights=(1,1)):
    '''
    3 inputs, second is inhibitory. Delayed syns
    '''
    snn = SNNModel()
    neu_out = Izhikevich(id=0, noise=noise)
    for i in range(1,4):
        neu = Spikes_at_will(id=i, refresh_time=rt, awaiting_time=interval*i)
        if i == 2:
            snn.add_synapse(Delayed_synapse(neu, neu_out, scale=scale, synaptic_limit=synaptic_limit, slow_variable_limit=synaptic_limit,
                                    lr=lr, d_lr=d_lr, max_delay=max_delay, b=b, tau=tau, slow_tau=slow_tau, forget_tau=forget_tau,
                                    delay=delay,weights=weights, inhibitory=True))
        else:
            snn.add_synapse(Delayed_synapse(neu, neu_out, scale=scale, synaptic_limit=synaptic_limit, slow_variable_limit=synaptic_limit,
                                    lr=lr, d_lr=d_lr, max_delay=max_delay, b=b, tau=tau, slow_tau=slow_tau, forget_tau=forget_tau,
                                    delay=delay,weights=weights, inhibitory=False))
    print(snn.show_config())
    snn.reload_graph()
    return snn


def syn_with_outer_cur(rt, aw, I=10, delay=1, d_lr=10):
    '''
    Generates two synapses - one actual that is to explore,
    and one fake, to provide output with DC
    '''
    model = SNNModel()
    neu0 = Spikes_at_will(refresh_time=rt, aw=aw)
    neu_out = Izhikevich()
    outer_cur = DirectNeuron(I=I)
    syn = Delayed_synapse(neu0, neu_out, w=1, delay=delay, max_delay=100, b=2, d_lr=d_lr)
    syn1 = NeverLearn(outer_cur, neu_out, w=1)
    model.add_neuron(neu0)
    model.add_neuron(neu_out)
    model.add_neuron(outer_cur)
    model.add_synapse(syn)
    model.add_synapse(syn1)
    model.define_output(1)
    model.reload_graph()
    return model


def single_delayed(rt, aw, delay=1, d_lr=10):
    model = SNNModel()
    neu0 = Spikes_at_will(refresh_time=rt, aw=aw)
    neu_out = Izhikevich()
    syn = Delayed_synapse(neu0, neu_out, w=1, delay=delay, max_delay=100, b=2, d_lr=d_lr, scale=7)
    model.add_neuron(neu0)
    model.add_neuron(neu_out)
    model.add_synapse(syn)
    model.define_output(1)
    model.reload_graph()
    return model