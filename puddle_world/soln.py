"""Find optimal Value and Policy functions for PuddleWorlds

"""

import numpy as np
import itertools as it


class EpsilonGreedyPolicy:
    """An Epsilon Greedy Policy wrt. a provided Q function"""

    def __init__(self, q, epsilon=0.0):
        """C-tor
        
        Args:
            q (numpy array): |S|x|A| Q-matrix
            epsilon (float): Probability of taking a random action
        """
        self.q = q
        self.epsilon = epsilon
        self.optimal_action_map = {s: np.argmax(q[s]) for s in range(q.shape[0])}

    @property
    def policy_mat(self):
        """Get stochastic policy matrix"""
        return np.array([self.prob_for_state(s) for s in range(self.q.shape[0])])

    def prob_for_state(self, s):
        """Get the action probability vector for a given state"""
        p = np.ones(self.q.shape[1]) * (self.epsilon) / self.q.shape[1]
        p[self.optimal_action_map[s]] += 1 - self.epsilon
        return p

    def act(self, s):
        if self.epsilon == 0.0:
            return self.optimal_action_map[s]
        else:
            return np.random.choice(
                np.arange(self.q.shape[1]), p=self.prob_for_state[s]
            )

    def get_rollouts(self, env, num_rollouts, *, max_episode_length=None):
        """Get rollouts of this policy in an environment
        
        Args:
            env (gym.Env): Environment to use
            num_rollouts: Number of rollouts to collect
            
        Returns:
            (list): List of collected (s, a) rollouts
        """
        rollouts = []
        for episode in it.count():
            rollout = []
            s = env.reset()
            for timestep in it.count():
                a = self.act(s)
                rollout.append((s, a))
                s, r, done, _ = env.step(a)
                if done:
                    break
                if timestep == max_episode_length:
                    break
            rollout.append((s, None))
            rollouts.append(rollout)
            if episode == num_rollouts:
                break

        return rollouts


def policy_evaluation(env, policy, discount=1.0, tolerance=1e-6):
    """Determine the value function of the given policy
    
    Args:
        env (PuddleWorldEnv): Environment to evaluate against
        policy (numpy array): |S|x|A| stochastic policy matrix
        
        discount (float): Discount factor
        tolerance (float): State value convergence threshold
    
    Returns:
        (numpy array): |S| state value vector
    """
    v_pi = np.zeros(env.num_states)

    for _iteration in it.count():
        delta = 0

        for s in range(env.num_states):
            v = v_pi[s]
            v_pi[s] = np.sum(
                [
                    policy[s, a]
                    * np.sum(
                        [
                            env.transition_matrix[s, a, s2]
                            * (env.reward(s2) + discount * v_pi[s2])
                            for s2 in range(env.num_states)
                        ]
                    )
                    for a in range(env.num_actions)
                ]
            )
            delta = max(delta, np.abs(v - v_pi[s]))

        if delta < tolerance:
            break

    return v_pi


def value_iteration(env, discount=1.0, tolerance=1e-6):
    """Value iteration to find the optimal value function"""

    value_fn = np.zeros((env.num_states))

    for _iter in it.count():
        delta = 0
        for s in range(env.num_states):
            v = value_fn[s]
            value_fn[s] = np.max(
                [
                    np.sum(
                        [
                            env.transition_matrix[s, a, s2]
                            * (env.reward(s2) + discount * value_fn[s2])
                            for s2 in range(env.num_states)
                        ]
                    )
                    for a in range(env.num_actions)
                ]
            )
            delta = max(delta, np.abs(v - value_fn[s]))

        # Check value function convergence
        if delta < tolerance:
            break

    return value_fn


def q_from_v(v_star, env, discount=1.0):
    """Find Q* given V*"""

    q_star = np.zeros((env.num_states, env.num_actions))

    for s, a, s2 in it.product(
        range(env.num_states), range(env.num_actions), range(env.num_states)
    ):
        q_star[s, a] += env.transition_matrix[s, a, s2] * (
            env.reward(s2) + discount * v_star[s2]
        )

    return q_star


def demo():
    """Demonstrate this module"""

    import matplotlib.pyplot as plt
    from puddle_world.envs import PuddleWorldEnv, CanonicalPuddleWorldEnv

    # Find the Optimal Q, V functions for the Canonical Puddle World Environment
    env = CanonicalPuddleWorldEnv(mode="dry")
    # env = PuddleWorldEnv(mode="dry", width=10, height=5)
    print(env._ascii())

    v_star = value_iteration(env, 1.0)
    q_star = q_from_v(v_star, env)
    pi_star = np.argmax(q_star, axis=1)
    pi_star_mat = np.zeros_like(q_star)
    for s in range(q_star.shape[0]):
        pi_star_mat[s, pi_star[s]] = 1.0
    v_pi = policy_evaluation(env, pi_star_mat)

    assert np.allclose(
        v_star, v_pi
    ), "There seems to be something wrong with Policy Evaluation or Value Iteration"

    # Plot V*
    plt.figure()
    plt.set_cmap("OrRd_r")
    plt.imshow(v_star.reshape((env.height, env.width)))
    for (j, i), val in np.ndenumerate(v_star.reshape((env.height, env.width))):
        plt.text(i, j, "{:.2f}".format(val), ha="center", va="center")
    plt.title("V*(s)")

    # Plot Q*
    ax, figs = plt.subplots(1, 4, sharey=True, figsize=(9, 3))
    for ai, a in enumerate(env.actions.keys()):
        plt.sca(figs[ai])
        plt.imshow(q_star[:, ai].reshape((env.height, env.width)))
        for (j, i), val in np.ndenumerate(
            q_star[:, ai].reshape((env.height, env.width))
        ):
            plt.text(i, j, "{:.2f}".format(val), ha="center", va="center")
        plt.xlabel("Action {}".format(a))
    plt.tight_layout()
    plt.suptitle("Q*(s, a)")

    # Plot pi*
    plt.figure()
    plt.imshow(v_star.reshape((env.height, env.width)))
    for (j, i), a in np.ndenumerate(pi_star.reshape((env.height, env.width))):
        plt.text(
            i,
            j,
            env.action_symbols[list(env.actions.keys())[a]],
            ha="center",
            va="center",
        )
    plt.title("π*(s)")

    plt.show()
    print("Done")


if __name__ == "__main__":
    demo()
