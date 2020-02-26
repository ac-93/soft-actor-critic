import numpy as np
import os
import tensorflow as tf

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    init_fn = tf.keras.initializers.Orthogonal(1.0)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, kernel_initializer=init_fn)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=init_fn)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""
def gumbel_policy(x, act_dim, hidden_sizes, activation):

    # policy network outputs
    net = mlp(x, list(hidden_sizes), activation, activation)
    logits = tf.layers.dense(net, act_dim, activation='linear')

    # action and log action probabilites (log_softmax covers numerical problems)
    action_probs = tf.nn.softmax([logits], axis=-1)
    log_action_probs = tf.nn.log_softmax([logits], axis=-1)

    # policy with no noise
    mu = tf.argmax(logits, axis=-1)

    # add gumbel noise to action distributions
    temperature = tf.convert_to_tensor(1.0) # 0 --> argmax, inf --> uniform
    uniform_noise = tf.random_uniform(shape=tf.shape(logits),
                                      minval=np.finfo(np.float32).tiny, # (0,1) range
                                      maxval=1.)
    gumbel_noise  = -tf.log(-tf.log(uniform_noise))
    noisy_logits  = logits + gumbel_noise
    pi_dist = tf.nn.softmax(noisy_logits / temperature[..., tf.newaxis])

    # dont use tf.dist.relaxedCategorical for log_prob, seems to give wrong results
    logp_pi = -tf.reduce_sum(-pi_dist * tf.nn.log_softmax(logits, axis=-1), axis=1)

    return mu, pi_dist, logp_pi

"""
Actor-Critics
"""
def a_out_mlp_actor_critic(x, a, hidden_sizes=[400,300], activation=tf.nn.relu, policy=gumbel_policy):

    act_dim = a.shape.as_list()[-1]

    with tf.variable_scope('pi'):
        mu, pi_dist, logp_pi = policy(x, act_dim, hidden_sizes, activation)

    # vfs
    with tf.variable_scope('q1'):
        q1    = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        q1_a  = tf.reduce_sum(tf.multiply(q1, a), axis=1)
    with tf.variable_scope('q2'):
        q2    = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        q2_a  = tf.reduce_sum(tf.multiply(q2, a), axis=1)

    return mu, pi_dist, logp_pi, q1_a, q2_a


def a_in_mlp_actor_critic(x, a, hidden_sizes=[400,300], activation=tf.nn.relu, policy=gumbel_policy):

    act_dim = a.shape.as_list()[-1]

    with tf.variable_scope('pi'):
        mu, pi_dist, logp_pi = policy(x, act_dim, hidden_sizes, activation)

    # vfs
    with tf.variable_scope('q1'):
        q1_a  = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    with tf.variable_scope('q2'):
        q2_a  = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    return mu, pi_dist, logp_pi, q1_a, q2_a
