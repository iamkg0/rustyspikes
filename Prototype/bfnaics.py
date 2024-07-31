from model import *
from neurons import *
from synaptics import *
from utils import *
from protocols import *
from shallow_models import *
from vis_functions import *
import pandas as pd
res = .1
counter = 0

cfg = {
    'lr': (.05, ),
    'learning_time': 100000,
    'test_time': 10000,
    'num_inp_neu': 5,
    'scale': (4,5,6,7,8,9,10,11),
    'aw': (1,2,3,4,5,6,7,8),
    'l_rule': ('t_stdp', 'pair_stdp')
    }
'''
cfg = {
    'lr': (.01, ),
    'learning_time': 100000,
    'test_time': 10000,
    'num_inp_neu': (10, ),
    'scale': (8, ),
    'aw': (3, ),
    'l_rule': ('pair_stdp', 't_stdp' )
    }
'''
def pipe(cfg, counter):
    out = [[] for i in cfg]
    for i in range(3):
        out.append([])
    weights = []
    num_input = cfg["num_inp_neu"]
    for lr in cfg['lr']:
        for scale in cfg['scale']:
            for aw in cfg['aw']:
                for l_rule in cfg['l_rule']:
                    awaiting = list(range(0, aw*num_input, aw))
                    print(f'Attempt {counter} with lr={lr}, scale={scale}, awaiting_time={aw}, l_rule={l_rule}, num_input={num_input}:')
                    snn = bfnaics_24_model(num_input=num_input, rt=100, scale=scale)
                    snn, gatherer, familiar, new = bfnaics24(snn, awaiting, lr=lr, l_rule=l_rule,
                                                    learning_time=cfg['learning_time'],
                                                    test_time=cfg['test_time'], draw_stats=False)
                    out[0].append(lr)
                    out[1].append(cfg['learning_time'])
                    out[2].append(cfg['test_time'])
                    out[3].append(num_input)
                    out[4].append(scale)
                    out[5].append(aw)
                    out[6].append(l_rule)
                    out[7].append(familiar)
                    out[8].append(new)
                    out[9].append(familiar-new)
                    counter += 1
                    cur_model_weights = []
                    for w in snn.syn_by_edge:
                        cur_model_weights.append(snn.syn_by_edge[w].get_weight())
                    weights.append(cur_model_weights)
    print(out)
    out = pd.DataFrame(out).transpose()
    out.columns = ['lr', 'l_time', 'test_time', 'num_input', 'scale', 'awaiting_time', 'l_rule', 'resp_familiar', 'resp_new', 'diff']
    out.to_csv(f'bfnaics24_{num_input}_neus.csv')
    np.save(f'bfnaics24_{num_input}_neus', np.array(weights))
    
    return out, np.array(weights), counter



neus_nums = (3,5,7,10)
for i in neus_nums:
    cfg['num_inp_neu'] = i
    data, weights, counter = pipe(cfg, counter)

plt.bar(list(range(10)), weights[0])
plt.show()
print(weights)