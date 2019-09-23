import sys, os
par_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(par_path)

import numpy as np
import time
import gym

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from image_utils import load_json_obj
from spinup_utils.logx import *

from sac_image_algo.sac import *

sess = tf.compat.v1.Session()

# Load model and config
model_dir = '/home/alexc/Documents/drl_algos/saved_models/sac_image_CarRacing-v0/sac_image_CarRacing-v0_s0'
model_name = 'simple_save8'
model = restore_tf_graph(sess=sess, fpath=os.path.join(model_dir, model_name))
x_ph = model['x']
pi = model['pi']
mu = model['mu']

config = load_json_obj(os.path.join(model_dir, 'config'))

test_env = gym.make(config['rl_params']['env_name'])
obs_dim = config['network_params']['input_dims']
test_state_buffer = StateBuffer(m=obs_dim[2])
max_ep_len = config['rl_params']['max_ep_len']


def get_action(state, deterministic=False):
    state = state.astype('float32') / 255.
    act_op = mu if deterministic else pi
    return sess.run(act_op, feed_dict={x_ph: [state]})[0]

def reset(env, state_buffer):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    o = process_observation(o, obs_dim)
    state = state_buffer.init_state(init_obs=o)
    return o, r, d, ep_ret, ep_len, state


def test_agent(n=10, render=True):
    for j in range(n):
        o, r, d, ep_ret, ep_len, test_state = reset(test_env, test_state_buffer)

        if render: test_env.render()

        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            o, r, d, _ = test_env.step(get_action(test_state, True))
            o = process_observation(o, obs_dim)
            test_state = test_state_buffer.append_state(o)
            ep_ret += r
            ep_len += 1

            if render: test_env.render()

        if render: test_env.close()



test_agent(n=10, render=True)
