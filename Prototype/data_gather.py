from rustyspikes import *
from delayed_paper import *
np.random.seed(42)

log_path = r'C:\Users\iamkg0\Desktop\projects\rustyspikes\Prototype\logs\testing'
log_name = 'test_1.csv'
log = LogHandler(path=log_path, filename=log_name)


# PROTOCOL CONFIGURATION

config = {
    'input_size': [5, 6, 7, 8],
    'aw_time': [4, 4.5, 5, 5.5, 6],
    'sample_time': [150],
    'lr': [.01],
    'runs': [250],
    'd_lr': [1],
    'scale': [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
    'rt': [250],
    'num_rand_patterns': [3, 5, 7],
    'synaptic_limit': [1, None],
    'slow_tau': [100],
    'forget_tau': [100],
    'b': [7.5, 7.6, 7.7],
    'max_delay': [100],
    'learning_rule': ['pair_stdp', 't_stdp', 'strdp', 'delayed'],
    'noise': [0.7, 1, 1.3, 1.6, 1.9],
    'weights': [(.2, .4), (.2, .6), (.4, .6), (.4, .8), (1, 1)],
}
del_w_to_one = True

num_tries = 1
for i in config:
    num_tries *= len(config[i])

patterns, num_inputs, num_patterns = sampler(num_inputs=config['input_size'][0], num_rand_patterns=config['num_rand_patterns'][0])
log.create()
log.define_cols(('input_size', 'aw_time', 'sample_time', 'lr', 'runs', 'd_lr', 'scale', 'rt',
                            'num_patterns', 'synaptic_limit', 'slow_var_limit', 'slow_tau', 'forget_tau',
                            'spikes (0 is gt)', 'b', 'max_delay', 'learning_rule', 'noise', 'init_weights'))
log.comment([*patterns.values()])
patterns



attempt_num = 0
cfgs = cfg_slicer(config)
for cfg in cfgs:

    attempt_num += 1
    print('\n Calculation in progress...')
    print(f'Run {attempt_num}/{num_tries}')

    if cfg['learning_rule'] == 'delayed':
        delayed = True
    else:
        delayed = False
    model = one_neu_dynamics(num_input=cfg['input_size'], scale=cfg['scale'], rt=cfg['rt'], aw=cfg['aw_time'], interval=cfg['aw_time'],
                            learning_rule=cfg['learning_rule'], delayed=delayed,
                            lr=cfg['lr'], tau=30, d_lr=.1, synaptic_limit=cfg['synaptic_limit'],
                            slow_variable_limit=cfg['synaptic_limit'], max_delay=cfg['max_delay'], b=cfg['b'],
                            slow_tau=cfg['slow_tau'], forget_tau=cfg['forget_tau'], delay=0,
                            noise=cfg['noise'], weights=cfg['weights'], stick_del_w_to_one=del_w_to_one)
    
    # TRAIN
    model, *_ = run_protocol(model, sampler=patterns, sample_time=cfg['sample_time'], interval=cfg['aw_time'],
                    runs=cfg['runs'], lr=cfg['lr'], d_lr=cfg['d_lr'], test=False,
                    freeze_delays=False, gather_data=False, plot=False, plast_type=cfg['learning_rule'],
                    return_gatherer=False, gather_delays=False, logger=None, init_weights=cfg['weights'],
                    stick_del_w_to_one=del_w_to_one)

    # TEST
    model, *_ = run_protocol(model, sampler=patterns, sample_time=cfg['sample_time'], interval=cfg['aw_time'],
                    runs=cfg['runs'], lr=cfg['lr'], d_lr=cfg['d_lr'], test=True,
                    freeze_delays=False, gather_data=False, plot=False, plast_type=cfg['learning_rule'],
                    return_gatherer=False, gather_delays=False, logger=log, init_weights=cfg['weights'],
                    stick_del_w_to_one=del_w_to_one)
    


print('Done!')