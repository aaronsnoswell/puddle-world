"""Implements MaxEnt IRL for PuddleWorldEnv

"""

import gym
import warnings
import numpy as np
import itertools as it


def logadd(*args):
    """Add several numbers in log-space"""
    m = max(*args)
    # Catch case when all args are negative infinity to avoid np.nan
    if np.isneginf(m):
        return m
    return m + np.log(np.sum([np.exp(a - m) for a in args]))


def marginal_probs(env, state_rewards, max_path_length):
    """Compute marginal probabilities of taking action 'a' from state 's' at time 't'
    
    Args:
        env (PuddleWorldEnv): Environment to consider
        state_rewards (numpy array): State reward vector
        max_path_length (int): Maximum path length to consider
    
    Returns:
        (numpy array): |S| x (max_path_length) forward message passing partition array
        (numpy array): |S| x (max_path_length) backward message passing partition array
        (float): Partition value
        (numpy array): |S|x|A|x(max_path_length-1) array containing p_t(S, a) under
            the given reward function and the Maximum Entropy policy assumption
    """

    # Compute forward message
    # N.b. order of iteration is important - must sweep times first
    alpha = np.zeros((env.num_states, max_path_length))
    alpha[:, 0] = env.start_state_distribution
    for t in range(max_path_length - 1):
        for s, a, s2 in env.sas_iter:
            if s in env.terminal_states:
                continue
            alpha[s2, t + 1] = alpha[s2, t + 1] + (
                alpha[s, t] * env.transition_matrix[s, a, s2] * np.exp(state_rewards[s])
            )

    # Compute backward message
    # N.b. order of iteration is important - must sweep times first
    beta = np.zeros((env.num_states, max_path_length))
    beta[:, 0] = 1.0
    for t in range(max_path_length - 1):
        for s, a, s2 in env.sas_iter:
            if s in env.terminal_states:
                beta[s, t + 1] = 1.0
            else:
                beta[s, t + 1] += (
                    env.transition_matrix[s, a, s2]
                    * np.exp(state_rewards[s])
                    * beta[s2, t]
                )

    z_theta = np.sum(alpha)
    p_s_a_t = np.zeros((env.num_states, env.num_actions, max_path_length))
    maxt = 0
    maxl = 0
    for t in range(max_path_length + 1):
        print("T is {}".format(t), p_s_a_t.shape)
        for l in range(max_path_length - (t + 1)):
            for s, a, s2 in env.sas_iter:
                maxt = max(maxt, t)
                maxl = max(maxl, l)
                p_s_a_t[s, a, t] += (
                    alpha[s, t]
                    * env.transition_matrix[s, a, s2]
                    * np.exp(state_rewards[s])
                    * beta[s2, l]
                )

    # Add partition contribution
    p_s_a_t /= z_theta

    return alpha, beta, z_theta, p_s_a_t


class OverloadRewardEnv(gym.Env):
    """An environment that overloads the reward function of another environment"""

    def __init__(self, env, state_rewards):
        """C-tor"""
        self._env = env
        self._state_rewards = state_rewards

    def step(self, action):
        obs, r, done, info = super().step()
        r_overload = self._state_rewards[obs]
        if info is None:
            info = {}
        info["original_reward"] = r
        return obs, r_overload, done, info


def demo():
    """Demonstrate this module"""

    import matplotlib.pyplot as plt
    from puddle_world.envs import (
        PuddleWorldEnv,
        CanonicalPuddleWorldEnv,
        SmallPuddleWorldEnv,
    )
    from puddle_world.soln import value_iteration, q_from_v, EpsilonGreedyPolicy

    # Find the Optimal Q, V functions for the Canonical Puddle World Environment
    env = SmallPuddleWorldEnv(mode="dry")

    print("Env:")
    print(env._ascii())
    print()

    # Solve the env
    v_star = value_iteration(env)
    q_star = q_from_v(v_star, env)
    pi_star = EpsilonGreedyPolicy(q_star)

    rollouts = pi_star.get_rollouts(env, 1000)

    # Find the expectation over feature vectors from each of the rollouts
    efv = np.zeros(env.num_states)
    for r in rollouts:
        for s, a in r[:-1]:
            efv[s] += 1
    efv /= len(rollouts)

    # Pick a starting reward function
    gt_state_rewards = np.array([env.reward(s) for s in range(env.num_states)])
    # state_rewards = np.zeros(env.num_states)
    state_rewards = gt_state_rewards.copy()

    print("Empirical EFV:")
    with np.printoptions(precision=2):
        print(efv.reshape(env.height, env.width))
    print()

    for max_path_length in range(2, 20 + 1):
        # for max_path_length in range(2, 3 + 1):

        print()
        print("===== Max Path Length: {}".format(max_path_length))

        # Find EFV under max-ent policy for current reward
        alpha, beta, z, mp = marginal_probs(env, state_rewards, max_path_length)
        efv_me = np.zeros(env.num_states)
        for s in range(env.num_states):
            efv_me[s] += np.sum(mp[s, :, 0 : max_path_length - 1])

        with np.printoptions(precision=2):
            print(efv_me.reshape(env.height, env.width))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2,
            2,
            sharex=True,
            sharey=True,
            figsize=(6, 6),
            dpi=300,
            gridspec_kw=dict(wspace=0.1, hspace=0.1),
        )
        # sns.set()
        plt.set_cmap("Greys_r")
        ax1.imshow(gt_state_rewards.reshape(env.height, env.width))
        for s in env.s_iter:
            y, x = env.s2yx(s)
            ax1.text(
                x,
                y,
                list(env.action_symbols.values())[pi_star.act(s)],
                ha="center",
                va="center",
                color="r",
            )
        ax1.set_title("GT Reward")
        ax2.imshow(state_rewards.reshape(env.height, env.width))
        ax2.set_title("Test Reward")
        ax3.imshow(efv.reshape(env.height, env.width))
        ax3.set_title("GT EFV")
        ax4.imshow(efv_me.reshape(env.height, env.width))
        ax4.set_title("ME EFV")
        # plt.show()
        plt.savefig("foo-mpl-{}".format(max_path_length))


if __name__ == "__main__":
    demo()
