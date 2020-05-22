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


def marginal_probs(env, state_rewards, max_path_length, *, min_path_length=1):
    """Compute marginal probabilities of taking action 'a' from state 's' at time 't'
    
    Args:
        env (PuddleWorldEnv): Environment to consider
        state_rewards (numpy array): State reward vector
        max_path_length (int): Maximum path length to consider
        
        min_path_length (int): Minimum path length to consider
    
    Returns:
        (numpy array): |S| x (max_path_length) forward message passing partition array
        (numpy array): |S| x (max_path_length) backward message passing partition array
        (float): Partition value
        (numpy array): |S|x|A|x(max_path_length-1) array containing p(a | s; t) under
            the given reward function and the Maximum Entropy policy assumption
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Get log transition matrix
        logt = np.log(env.transition_matrix)

        # Get log start state probabilities
        logp0 = np.log(env.start_state_distribution)

        # Prepare log message passing matrices
        logalpha = np.log(np.zeros((env.num_states, max_path_length)))
        logbeta = np.log(np.zeros((env.num_states, max_path_length)))
        logalpha[:, 0] = logp0
        logbeta[:, 0] = np.log(1.0)

        # Prepare log marginal s, a, t probabilities
        logm = np.log(np.zeros((env.num_states, env.num_actions, max_path_length - 1)))

    # Pass forward, backward messages
    for t in range(max_path_length - 1):
        for s, a, s2 in it.product(
            range(env.num_states), range(env.num_actions), range(env.num_states)
        ):
            logalpha[s2, t + 1] = logadd(
                logalpha[s2, t + 1], logalpha[s, t] + logt[s, a, s2] + state_rewards[s]
            )

            logbeta[s, t + 1] = logadd(
                logbeta[s, t + 1], logt[s, a, s2] + state_rewards[s] + logbeta[s2, t],
            )

    # Compute log partition
    logz = logadd(logalpha[:, min_path_length - 1 : max_path_length].flatten())

    # Compute marginal probabilities of choosing a state, action at a time
    p_s_a_t = np.zeros((env.num_states, env.num_actions, max_path_length - 1))
    for s, a, s2 in it.product(
        range(env.num_states), range(env.num_actions), range(env.num_states)
    ):
        for t in range(max_path_length - 1):
            lower_l = max(min_path_length - t - 2, 0)
            upper_l = max_path_length - t - 1
            for l in range(lower_l, upper_l):
                p_s_a_t[s, a, t] += np.exp(
                    logalpha[s, t]
                    + logt[s, a, s2]
                    + state_rewards[s]
                    + logbeta[s2, l]
                    - logz
                )

    # return efv
    return np.exp(logalpha), np.exp(logbeta), np.exp(logz), p_s_a_t


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
    from puddle_world.envs import PuddleWorldEnv, CanonicalPuddleWorldEnv
    from puddle_world.soln import value_iteration, q_from_v, EpsilonGreedyPolicy

    # Find the Optimal Q, V functions for the Canonical Puddle World Environment
    env = CanonicalPuddleWorldEnv(mode="dry")

    # Solve the env
    v_star = value_iteration(env)
    q_star = q_from_v(v_star, env)
    pi_star = EpsilonGreedyPolicy(q_star)

    rollouts = pi_star.get_rollouts(env, 100000)

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

    with np.printoptions(precision=2):
        print(efv.reshape(5, 5))

    for max_path_length in range(2, 20 + 1):

        # Find EFV under max-ent policy for current reward
        alpha, beta, z, mp = marginal_probs(env, state_rewards, max_path_length)
        efv_me = np.zeros(env.num_states)
        for s in range(env.num_states):
            efv_me[s] += np.sum(mp[s, :, 0 : max_path_length - 1])

        print("Max Path Length: {}".format(max_path_length))
        with np.printoptions(precision=2):
            print(efv_me.reshape(5, 5))

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
        ax1.imshow(gt_state_rewards.reshape(5, 5))
        ax1.set_title("GT Reward")
        ax2.imshow(state_rewards.reshape(5, 5))
        ax2.set_title("Test Reward")
        ax3.imshow(efv.reshape(5, 5))
        ax3.set_title("GT EFV")
        ax4.imshow(efv_me.reshape(5, 5))
        ax4.set_title("ME EFV")
        plt.savefig("foo-mpl-{}".format(max_path_length))
        # plt.show()


if __name__ == "__main__":
    demo()
