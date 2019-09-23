import numpy as np
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPS = 1e-8

def placeholder(dim=None):
    if isinstance(dim, (list,)):
        ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,*dim])
    else:
        ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))
    return ph

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def build_model(x,
                concat_policy=None,
                output_dim=None,
                input_dims=[100,100],
                conv_filters=(8, 16, 32, 32),
                dense_units=(512,),
                kernel_width=3,
                strides=1,
                pooling='max',
                pooling_width=2,
                pooling_strides=1,
                hidden_activation='relu',
                output_activation='linear',
                batch_norm=False,
                dropout=0.0
                ):

    num_conv_layers = len(conv_filters)
    num_dense_layers = len(dense_units)
    num_hidden_layers = num_conv_layers + num_dense_layers
    num_layers = num_hidden_layers + 1

    # Replicate default parameters across layers, if required
    pooling = (pooling,) * num_conv_layers
    pooling_width = (pooling_width,) * num_conv_layers
    pooling_strides = (pooling_strides,) * num_conv_layers
    hidden_activation = (hidden_activation,) * num_hidden_layers
    batch_norm = (batch_norm,) * num_layers
    dropout = (dropout,) * num_layers

    initializer = tf.compat.v1.initializers.variance_scaling(scale=2.0)

    # Convolutional base
    for i in range(num_conv_layers):

        x = tf.layers.conv2d(inputs=x,
                             filters=conv_filters[i],
                             kernel_size=(kernel_width[i], kernel_width[i]),
                             strides=(strides[i], strides[i]),
                             activation=hidden_activation[i],
                             kernel_initializer=initializer)

        if batch_norm[i]:
            x = tf.layers.batch_normalization(inputs=x)

        if pooling[i] == 'max':
            x = tf.layers.max_pooling2d(inputs=x,
                                        pool_size=(pooling_width[i], pooling_width[i]),
                                        strides=(pooling_strides[i], pooling_strides[i]))
        elif pooling[i] == 'avg':
            x = tf.layers.average_pooling2d(inputs=x,
                                        pool_size=(pooling_width[i], pooling_width[i]),
                                        strides=(pooling_strides[i], pooling_strides[i]))

    # Dense layers
    x = tf.layers.flatten(inputs=x)

    # concat the action or policy into the network just before fully connected layers
    if concat_policy is not None:
        x = tf.concat([x,concat_policy], axis=-1)

    for i, j in enumerate(range(num_conv_layers, num_hidden_layers)):

        x = tf.layers.dense(inputs=x,
                            units=dense_units[i],
                            activation=hidden_activation[j],
                            kernel_initializer=initializer)

        if dropout[j] > 0.0:
            x = tf.layers.dropout(inputs=x,
                                  rates=dropout[j])

    # Output layer
    if output_dim is not None:
        x = tf.layers.dense(inputs=x,
                                units=output_dim,
                                activation=output_activation,
                                kernel_initializer=initializer)

    return x

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
"""
def kl_policy(x, act_dim, network_params):

    # policy network outputs
    net = build_model(x, concat_policy=None, output_dim=None, **network_params)
    logits = tf.layers.dense(net, act_dim, activation='linear')

    # action and log action probabilites (log_softmax covers numerical problems)
    action_probs = tf.nn.softmax(logits, axis=-1)
    log_action_probs = tf.nn.log_softmax(logits, axis=-1)

    # policy with no noise
    mu = tf.argmax(logits, axis=-1)

    # polciy with noise
    policy_dist = tf.distributions.Categorical(logits=logits)
    pi = policy_dist.sample()

    pi_entropy = -tf.reduce_sum(action_probs * log_action_probs, axis=-1)

    onehot_pi = tf.one_hot(pi, depth=act_dim, axis=-1, dtype=tf.float32)

    return mu, pi, onehot_pi, pi_entropy, logits

def build_models(x, a, act_dim, network_params):

    # policy
    with tf.variable_scope('pi'):
        mu, pi, onehot_pi, pi_entropy, pi_logits = kl_policy(x, act_dim, network_params)

    # vfs
    with tf.variable_scope('q1'):
        q1_logits = build_model(x, concat_policy=None, output_dim=act_dim, **network_params)
        q1_a  = tf.reduce_sum(tf.multiply(q1_logits, a), axis=1)
        q1_pi = tf.reduce_sum(tf.multiply(q1_logits, onehot_pi), axis=1)

    with tf.variable_scope('q2'):
        q2_logits = build_model(x, concat_policy=None, output_dim=act_dim, **network_params)
        q2_a  = tf.reduce_sum(tf.multiply(q2_logits, a), axis=1)
        q2_pi = tf.reduce_sum(tf.multiply(q2_logits, onehot_pi), axis=1)


    return mu, pi, pi_entropy, pi_logits, q1_logits, q2_logits, q1_a, q2_a, q1_pi, q2_pi
