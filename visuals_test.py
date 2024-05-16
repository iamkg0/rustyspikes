from model import *
from neurons import *
from synaptics import *
from utils import *
from protocols import *
from shallow_models import *
from pyvis.network import Network

def pyvis_test(model):
    net = Network(notebook = True, cdn_resources = "remote",
                bgcolor = "#222222",
                font_color = "white",
                width = "1920px",
                height = "1080px")
    G = model.spit_for_pyvis()
    print(G)
    net.from_nx(G)
    net.show('graph.html')


model = m_v0()
pyvis_test(model)