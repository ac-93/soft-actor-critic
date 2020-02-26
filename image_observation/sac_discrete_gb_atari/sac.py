import sys, os
import numpy as np
import time
import gym
import tensorflow as tf
from spinup.utils.logx import EpochLogger

from common_utils import *
from core import *

# configure gpu use and supress tensorflow warnings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
tf_config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def sac(env_fn, logger_kwargs=dict(), network_params=dict(), rl_params=dict()):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # env params
    thresh          = rl_params['thresh']

    # control params
    seed            = rl_params['seed']
    epochs          = rl_params['epochs']
    steps_per_epoch = rl_params['steps_per_epoch']
    replay_size     = rl_params['replay_size']
    batch_size      = rl_params['batch_size']
    start_steps     = rl_params['start_steps']
    max_ep_len      = rl_params['max_ep_len']
    max_noop        = rl_params['max_noop']
    save_freq       = rl_params['save_freq']
    render          = rl_params['render']

    # rl params
    gamma           = rl_params['gamma']
    polyak          = rl_params['polyak']
    lr              = rl_params['lr']
    grad_clip_val   = rl_params['grad_clip_val']

    alpha                = rl_params['alpha']
    target_entropy_start = rl_params['target_entropy_start']
    target_entropy_stop  = rl_params['target_entropy_stop']
    target_entropy_steps = rl_params['target_entropy_steps']

    train_env, test_env = env_fn(), env_fn()
    obs_space = env.observation_space
    act_space = env.action_space

    tf.set_random_seed(seed)
    np.random.seed(seed)
    train_env.seed(seed)
    train_env.action_space.np_random.seed(seed)
    test_env.seed(seed)
    test_env.action_space.np_random.seed(seed)

    # get the size after resize
    obs_dim = network_params['input_dims']
    act_dim = act_space.n

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # init a state buffer for storing last m states
    train_state_buffer = StateBuffer(m=obs_dim[2])
    test_state_buffer  = StateBuffer(m=obs_dim[2])

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = placeholders(obs_dim, act_dim, obs_dim, None, None)

    # alpha and entropy setup
    max_target_entropy = tf.log(tf.cast(act_dim, tf.float32))
    target_entropy_prop_ph =  tf.placeholder(dtype=tf.float32, shape=())
    target_entropy = max_target_entropy * target_entropy_prop_ph

    log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)

    if alpha == 'auto': # auto tune alpha
        alpha = tf.exp(log_alpha)
    else: # fixed alpha
        alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=alpha)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, pi_logits, q1_logits, q2_logits, q1_a, q2_a, q1_pi, q2_pi = build_models(x_ph, a_ph, act_dim, network_params)

    # Target value network
    with tf.variable_scope('target'):
        _, _, logp_pi_targ, _, _, _,  _, _, q1_pi_targ, q2_pi_targ = build_models(x2_ph, a_ph, act_dim, network_params)

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in
                       ['log_alpha', 'main/pi', 'main/q1', 'main/q2', 'main'])
    print(('\nNumber of parameters: \t alpha: %d, \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t total: %d\n')%var_counts)

    # Min Double-Q: (check the logp_pi bit)
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    min_q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)

    # Targets for Q regression
    q_backup = r_ph + gamma*(1-d_ph)*tf.stop_gradient(min_q_pi_targ - alpha * logp_pi_targ)

    # critic losses
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_a)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_a)**2)
    value_loss = q1_loss + q2_loss

    # actor loss
    pi_loss = tf.reduce_mean(alpha*logp_pi - min_q_pi)

    # alpha loss for temperature parameter
    alpha_backup = tf.stop_gradient(logp_pi + target_entropy)
    alpha_loss   = -tf.reduce_mean(log_alpha * alpha_backup)

    # Policy train op
    # (has to be separate from value train op, because q1_logits appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-04)
    if grad_clip_val is not None:
        gvs = pi_optimizer.compute_gradients(pi_loss,  var_list=get_vars('main/pi'))
        capped_gvs = [(ClipIfNotNone(grad, grad_clip_val), var) for grad, var in gvs]
        train_pi_op = pi_optimizer.apply_gradients(capped_gvs)
    else:
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-04)
    with tf.control_dependencies([train_pi_op]):
        if grad_clip_val is not None:
            gvs = value_optimizer.compute_gradients(value_loss, var_list=get_vars('main/q'))
            capped_gvs = [(ClipIfNotNone(grad, grad_clip_val), var) for grad, var in gvs]
            train_value_op = value_optimizer.apply_gradients(capped_gvs)
        else:
            train_value_op = value_optimizer.minimize(value_loss, var_list=get_vars('main/q'))

    # Alpha train op
    alpha_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-04)
    with tf.control_dependencies([train_value_op]):
        train_alpha_op = alpha_optimizer.minimize(alpha_loss, var_list=get_vars('log_alpha'))

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops = [pi_loss, q1_loss, q2_loss, q1_a, q2_a,
                logp_pi, target_entropy,
                alpha_loss, alpha,
                train_pi_op, train_value_op, train_alpha_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                outputs={'mu': mu, 'pi': pi, 'q1': q1_a, 'q2': q2_a})

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
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
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
                    a = get_action(test_state, True)

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
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # ================== Main training Loop  ==================

    start_time = time.time()
    o, r, d, ep_ret, ep_len, old_lives, state = reset(train_env, train_state_buffer)
    total_steps = steps_per_epoch * epochs

    target_entropy_prop = linear_anneal(current_step=0, start=target_entropy_start, stop=target_entropy_stop, steps=target_entropy_steps)
    save_iter = 0
    terminal_life_lost = False

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # press fire to start
        if terminal_life_lost:
            a = 1
        else:
            if t > start_steps:
                a = get_action(state)
            else:
                a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        o2        = process_image_observation(o2, obs_dim, thresh)
        r         = process_reward(r)
        one_hot_a = process_action(a, act_dim)

        next_state = train_state_buffer.append_state(o2)

        ep_ret += r
        ep_len += 1

        if train_env.ale.lives() < old_lives:
            old_lives = train_env.ale.lives()
            terminal_life_lost = True
        else:
            terminal_life_lost = False

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(state, one_hot_a, r, next_state, terminal_life_lost)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2
        state = next_state

        if d or (ep_len == max_ep_len):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph:  batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph:  batch['acts'],
                             r_ph:  batch['rews'],
                             d_ph:  batch['done'],
                             target_entropy_prop_ph: target_entropy_prop
                            }
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0],
                             LossQ1=outs[1],    LossQ2=outs[2],
                             Q1Vals=outs[3],    Q2Vals=outs[4],
                             LogPi=outs[5], TargEntropy=outs[6],
                             LossAlpha=outs[7], Alpha=outs[8])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len, old_lives, state = reset(train_env, train_state_buffer)


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # update target entropy every epoch
            target_entropy_prop = linear_anneal(current_step=t, start=target_entropy_start, stop=target_entropy_stop, steps=target_entropy_steps)

            # Save model
            if save_freq is not None:
                if (epoch % save_freq == 0) or (epoch == epochs-1):
                    print('Saving...')
                    logger.save_state({'env': env},  itr=save_iter)
                    save_iter+=1

            # Test the performance of the deterministic version of the agent.
            test_agent(n=5, render=render)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', average_only=True)
            logger.log_tabular('TargEntropy', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':

    from spinup.utils.run_utils import setup_logger_kwargs

    network_params = {
        'input_dims':[84,84,4],
        'conv_filters':(32, 64, 64, 1024),
        'kernel_width':(8,4,3,7),
        'strides':(4,2,1,1),
        'pooling':'none',
        'pooling_width':2,
        'pooling_strides':1,
        'dense_units':(),
        'hidden_activation':'relu',
        'output_activation':'linear',
        'batch_norm':False,
        'dropout':0.0
    }

    rl_params = {
        # env params
        'env_name':'BreakoutDeterministic-v4',
        # 'env_name':'PongDeterministic-v4',
        'thresh':True,

        # control params
        'seed':int(0),
        'epochs':int(100),
        'steps_per_epoch':10000,
        'replay_size':int(4e5),
        'batch_size':32,
        'start_steps':50000,
        'max_ep_len':18000,
        'max_noop':10,
        'save_freq':5,
        'render':True,

        # rl params
        'gamma':0.99,
        'polyak':0.995,
        'lr':0.00025,
        'grad_clip_val':5.0,

        # entropy params
        'alpha': 'auto',
        'target_entropy_start':0.5, # proportion of max_entropy
        'target_entropy_stop':0.5,
        'target_entropy_steps':1e6,
    }

    saved_model_dir = '../../saved_models'
    logger_kwargs = setup_logger_kwargs(exp_name='sac_discrete_gb_atari_' + rl_params['env_name'], seed=rl_params['seed'], data_dir=saved_model_dir, datestamp=False)

    env = gym.make(rl_params['env_name'])

    # avoids crash when later rendering the environment
    if rl_params['render']:
        test_env(lambda:env)

    sac(lambda:env, logger_kwargs=logger_kwargs,
                    network_params=network_params,
                    rl_params=rl_params)
