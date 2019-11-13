import numpy as np
import cv2
import time
import tensorflow as tf

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_dim = obs_dim
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.uint8)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        # return normalised float observations
        return dict(obs1=self.obs1_buf[idxs].astype('float32') / 255.,
                    obs2=self.obs2_buf[idxs].astype('float32') / 255.,
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""
Store the observations in ring buffer type array of size m
"""
class StateBuffer:
    def __init__(self,m):
        self.m = m

    def init_state(self, init_obs):
        self.current_state = np.stack([init_obs]*self.m, axis=2)
        return self.current_state

    def append_state(self, obs):
        new_state = np.concatenate( (self.current_state, obs[...,np.newaxis]), axis=2)
        self.current_state = new_state[:,:,1:]
        return self.current_state

"""
Linear annealing from start to stop value based on current step and max_steps
"""
def linear_anneal(current_step, start=0.1, stop=1.0, steps=1e6):
    if current_step<=steps:
        eps = stop + (start - stop) * (1 - current_step/steps)
    else:
        eps=stop
    return eps

"""
Run a quick test of the environment, needed due to error when rendering
atari, for some reason doing this before hand fixes it...
"""
def test_env(env_fn, num_steps=25):
    env = env_fn()
    action_space = env.action_space
    env.reset()
    for i in range(num_steps):
        action = action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.05)
    env.close()

"""
Process the observation into grayscale, uint8 format for saving memory in replay buffer.
Possibly threshold the image to speed up training
"""
def process_image_observation(observation, obs_dim, thresh=False):
    if list(observation.shape) != obs_dim:
        # Convert to gray scale
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        # Add channel axis
        observation = observation[..., np.newaxis]
        # Resize to specified dims
        observation = cv2.resize(observation, (obs_dim[0], obs_dim[1]), interpolation=cv2.INTER_AREA)
        # Add channel axis
        observation = observation[..., np.newaxis]

        if thresh:
            ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)

        observation = observation.squeeze() # remove channel axis

    return observation

"""
Process discrete action into one hot vector
"""
def process_action(a, act_dim):
    one_hot_a = np.eye(act_dim)[a]
    return one_hot_a

"""
Process the reward by clipping as per mnih et al
"""
def process_reward(reward):
    return np.clip(reward, -1., 1.)

"""
Clip gradient whilst handling None error
"""
def ClipIfNotNone(grad, grad_clip_val):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -grad_clip_val, grad_clip_val)
