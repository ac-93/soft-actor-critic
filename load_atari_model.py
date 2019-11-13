import sys, os
import numpy as np
import time
import gym
import tensorflow as tf

from spinup.utils.logx import *
from image_observation.sac_discrete_kl_atari.common_utils import *

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_json_obj(name):
    with open(name + '.json', 'r') as fp:
        return json.load(fp)

def load_and_test_model(model_dir, model_save_name):

    sess = tf.compat.v1.Session(config=tf_config)

    model = restore_tf_graph(sess=sess, fpath=os.path.join(model_dir, model_save_name))
    config = load_json_obj(os.path.join(model_dir, 'config'))
    test_env = gym.make(config['rl_params']['env_name'])

    x_ph = model['x']
    mu = model['mu']
    pi = model['pi']

    obs_dim = config['network_params']['input_dims']
    test_state_buffer = StateBuffer(m=obs_dim[2])
    max_ep_len = config['rl_params']['max_ep_len']
    max_noop = config['rl_params']['max_noop']
    thresh = config['rl_params']['thresh']

    def get_action(state, deterministic=False):
        state = state.astype('float32') / 255.
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: [state]})[0]

    def reset(env, state_buffer):
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # fire to start game and perform no-op for some frames to randomise start
        o, _, _, _ = env.step(1) # Fire action to start game
        for _ in range(np.random.randint(1, max_noop)):
                o, _, _, _ = env.step(0) # Action 'NOOP'

        o = process_image_observation(o, obs_dim, thresh)
        r = process_reward(r)
        old_lives = env.ale.lives()
        state = state_buffer.init_state(init_obs=o)
        return o, r, d, ep_ret, ep_len, old_lives, state

    def test_agent(n=10, render=True):
        global sess, mu, pi, q1, q2
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
                    a = get_action(test_state, False)

                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(a)
                o = process_image_observation(o, obs_dim, thresh)
                r = process_reward(r)
                test_state = test_state_buffer.append_state(o)
                ep_ret += r
                ep_len += 1

                if test_env.ale.lives() < test_old_lives:
                    test_old_lives = test_env.ale.lives()
                    terminal_life_lost_test = True
                else:
                    terminal_life_lost_test = False

                if render: test_env.render()

            if render: test_env.close()

            print('Ep Return: ', ep_ret)



    test_agent(n=5, render=True)
    test_env.close()

if __name__ == '__main__':
    model_dir = 'saved_models/sac_discrete_kl_atari_BreakoutDeterministic-v4/sac_discrete_kl_atari_BreakoutDeterministic-v4_s1/'
    model_save_name = 'simple_save46'
    load_and_test_model(model_dir, model_save_name)
