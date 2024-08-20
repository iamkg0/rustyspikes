from model import *
from neurons import *
from synaptics import *
from utils import *
from shallow_models import *
from vis_functions import *
from pyvis.network import Network
res = .1
plt.style.use(['dark_background'])


def test_prot_v0(shallow_model, time=1000):
    snn = shallow_model()
    snn.set_rule_to_all('t_stdp')
    snn.set_lr_to_all(.1)
    gatherer = Gatherer(snn)
    t = np.arange(int(time / res)) * res
    for i in t:
        snn.tick()
        gatherer.gather_stats()
    draw_stats_gatherer(*gatherer.get_stats(), t)
    return snn

def test_d_single(shallow_model, time=1000):
    delay = []
    dsnn = shallow_model
    dd = []
    gatherer = Gatherer(dsnn)
    t = np.arange(int(time/res)) * res
    for i in t:
        dsnn.tick()
        gatherer.gather_stats(gather_delay=False)
        delay.append(dsnn.syn_by_edge[0, 1].delay)
        dd.append(dsnn.syn_by_edge[0, 1].dd)
    #print(*gatherer.get_stats(), t)
    draw_stats_gatherer(*gatherer.get_stats(), t)
    return dsnn, delay, np.array(dd)

def test_delay_v0(shallow_model, time=1000):
    delay = []
    dsnn = shallow_model()
    dd = []
    #dsnn.set_rule_to_all('sophisticated_rule')
    gatherer = Gatherer(dsnn)
    t = np.arange(int(time/res)) * res
    for i in t:
        dsnn.tick()
        gatherer.gather_stats()
        delay.append(dsnn.syn_by_edge[0, 1].delay)
        dd.append(dsnn.syn_by_edge[0,1].dd)
    draw_stats_gatherer(*gatherer.get_stats(), t)
    return dsnn, delay, np.array(dd)

def test_delay_v1(model, time=1000):
    delay = [[] for i in model.show_config()['Neurons']]
    delay = delay[:-1]
    dd = [[] for i in model.show_config()['Synapses']]
    #dd = dd[:-1]
    model.set_rule_to_all(None)
    gatherer = Gatherer(model)
    t = np.arange(int(time/res)) * res
    for i in t:
        model.tick()
        #if model.neurons[3].get_spike_status():
            #print(i)
        gatherer.gather_stats()
        for j in range(len(delay)):
            delay[j].append(model.syn_by_edge[j, 3].delay)
        for d in range(len(dd)):
            dd[d].append(model.syn_by_edge[d, 3].dd)
    draw_stats_gatherer(*gatherer.get_stats(), t)
    return model, np.array(delay), np.array(dd)


def bfnaics24(model, aw_time, l_rule='pair_stdp', lr=.1, learning_time=1000, test_time=1000, draw_stats=True):
    for i in range(len(model.neurons) - 1):
        model.neurons[i].change_props(awaiting_time=aw_time[i])
    gatherer = Gatherer(model)
    model, gatherer = bfnaics24_train(model, gatherer, learning_time, l_rule, lr)
    gatherer.show_spike_stats()
    gatherer.drop_timer_and_spikes()
    model, gatherer = bfnaics24_test(model, gatherer, test_time)
    _, f_familiar = gatherer.show_spike_stats()
    gatherer.drop_timer_and_spikes()
    for i in range(len(model.neurons) - 1):
        model.neurons[i].change_props(awaiting_time=aw_time[-i-1])
    model, gatherer = bfnaics24_test(model, gatherer, test_time)
    _, f_new = gatherer.show_spike_stats()
    gatherer.drop_timer_and_spikes()
    time = learning_time + test_time*2
    t = np.arange(int(time / res)) * res
    if draw_stats:
        draw_stats_gatherer(*gatherer.get_stats(), t)
    return model, gatherer, f_familiar, f_new

def bfnaics24_train(model, gatherer, time, rule, lr):
    model.set_rule_to_all(rule)
    model.set_lr_to_all(lr)
    t = np.arange(int(time / res)) * res
    for i in t:
        model.tick()
        gatherer.gather_stats()
    return model, gatherer

def bfnaics24_test(model, gatherer, time):
    model.set_rule_to_all(None)
    t = np.arange(int(time / res))* res
    for i in t:
        model.tick()
        gatherer.gather_stats()
    return model, gatherer