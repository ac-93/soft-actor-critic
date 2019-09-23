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

from atari.dqn_algo.dqn import *
from atari.dqn_algo.dqn import StateBuffer

sess = tf.compat.v1.Session()

# Load model and config
# model_dir = '/home/alexc/Documents/drl_algos/saved_models/atari_dqn_BreakoutDeterministic-v4/atari_dqn_BreakoutDeterministic-v4_s123'
model_dir = '/home/alexc/Documents/drl_algos/saved_models/atari_sac_CarRacing-v0/atari_sac_CarRacing-v0_s123'
model_name = 'simple_save'
model = restore_tf_graph(sess=sess, fpath=os.path.join(model_dir, model_name))
x_ph = model['x']
pi = model['pi']

config = load_json_obj(os.path.join(model_dir, 'config'))

test_env = env = gym.make(config['rl_params']['env_name'])
obs_dim = config['network_params']['input_dims']
test_state_buffer = StateBuffer(m=obs_dim[2])
max_ep_len = config['rl_params']['max_ep_len']
test_act_noise = config['rl_params']['test_act_noise']
max_noop = config['rl_params']['max_noop']


def get_action(state, noise_scale):
    if np.random.random_sample() < noise_scale:
        a = env.action_space.sample()
    else:
        a = sess.run(pi, feed_dict={x_ph: [state]})[0]
    return a

def reset(env, state_buffer):
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # fire to start game and perform no-op for some frames to randomise start
    o, _, _, _ = env.step(1) # Fire action to start game
    for _ in range(np.random.randint(1, max_noop)):
            o, _, _, _ = env.step(0) # Action 'NOOP'

    o = process_observation(o, obs_dim)
    r = process_reward(r)
    old_lives = env.ale.lives()
    state = state_buffer.init_state(init_obs=o)
    return o, r, d, ep_ret, ep_len, old_lives, state


def test_agent(n=10, render=False):
    print('Testing...')
    for j in range(n):
        o, r, d, ep_ret, ep_len, test_old_lives, test_state = reset(test_env, test_state_buffer)
        terminal_life_lost_test = False

        if render: test_env.render()

        while not(d or (ep_len == max_ep_len)):

            # start by firing
            if terminal_life_lost_test:
                a = 1
            else:
                # Take  lower variance actions at test(noise_scale=0.05)
                # a = get_action(test_state, test_act_noise)
                a = get_action(test_state, 0)

            # print(env.unwrapped.get_action_meanings()[a])

            o, r, d, _ = test_env.step(a)
            o = process_observation(o, obs_dim)
            r = process_reward(r)
            test_state = test_state_buffer.append_state(o)

            ep_ret += r
            ep_len += 1

            if test_env.ale.lives() < test_old_lives:
                test_old_lives = test_env.ale.lives()
                terminal_life_lost_test = True
            else:
                terminal_life_lost_test = False

            if render: test_env.render(), time.sleep(0.05)

    if render: test_env.close()



test_agent(n=10, render=True)
