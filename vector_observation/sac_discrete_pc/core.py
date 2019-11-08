import numpy as np
import os
import tensorflow as tf

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Policies
"""
def kl_policy(x, act_dim, hidden_sizes, activation):

    # policy network outputs
    net = mlp(x, list(hidden_sizes), activation, activation)
    logits = tf.layers.dense(net, act_dim, activation='linear')

    # action and log action probabilites (log_softmax covers numerical problems)
    action_probs     = tf.nn.softmax(logits, axis=-1)
    log_action_probs = tf.nn.log_softmax(logits, axis=-1)

    # policy with no noise
    mu = tf.argmax(logits, axis=-1)

    # polciy with noise
    policy_dist = tf.distributions.Categorical(logits=logits)
    pi = policy_dist.sample()

    return mu, pi, action_probs, log_action_probs

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=[400,300], activation=tf.nn.relu, policy=kl_policy):

    act_dim = a.shape.as_list()[-1]

    with tf.variable_scope('pi'):
        mu, pi, action_probs, log_action_probs = kl_policy(x, act_dim, hidden_sizes, activation)

    # vfs
    with tf.variable_scope('q1'):
        q1_logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        q1_a  = tf.reduce_sum(tf.multiply(q1_logits, a), axis=1)

    with tf.variable_scope('q2'):
        q2_logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
        q2_a  = tf.reduce_sum(tf.multiply(q2_logits, a), axis=1)

    return mu, pi, action_probs, log_action_probs, q1_logits, q2_logits, q1_a, q2_a
