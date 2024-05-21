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
    snn.set_rule_to_all('t_stdp_forget')
    snn.set_lr_to_all(.1)
    gatherer = Gatherer(snn)
    t = np.arange(int(time / res)) * res
    for i in t:
        snn.tick()
        gatherer.gather_stats()
    draw_stats_gatherer(*gatherer.get_stats(), t)
    return snn

def test_delay_v0(shallow_model, time=1000):
    delay = []
    dsnn = shallow_model()
    dsnn.set_rule_to_all(None)
    gatherer = Gatherer(dsnn)
    t = np.arange(int(time/res)) * res
    for i in t:
        dsnn.tick()
        gatherer.gather_stats()
        delay.append(dsnn.syn_by_edge[0, 1].delay)
    draw_stats_gatherer(*gatherer.get_stats(), t)
    return dsnn, delay

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
