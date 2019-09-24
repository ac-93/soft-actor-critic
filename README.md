# soft-actor-critic

### Implentations of Soft Actor Critic (SAC) algorithms from:

1: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018 https://arxiv.org/abs/1801.01290

2: Soft Actor-Critic Algorithms and Applications, Haarnoja et al, 2019, https://arxiv.org/abs/1812.05905

### Based on the implementations given in Spinningup

https://spinningup.openai.com/en/latest/algorithms/sac.html

Two different methods given for using SAC with discrete action spaces. 

* sac_discrete_gb uses the Gumbel Softmax distribtuion to reparameterize the discrete action space. This keeps algorithm similar to the original SAC implementation for continuous action spaces.
   
* sac_discrete_kl avoids reparmeterisation and calculates the entropy and KL divergence from the discrete actions given by the policy network.
   
A version of the continuous and discrete_kl algorithms that work with image observations such as the atari
gym environments are under sac_cont_image and sac_discrete_kl_atari.

I've not yet done too much testing so it is possible that there are some problems with the implementations...
