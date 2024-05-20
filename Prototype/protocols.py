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