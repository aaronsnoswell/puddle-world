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

    def act(self, s):
        if self.epsilon == 0.0:
            return self.optimal_action_map[s]
        else:
            p = np.ones(self.q.shape[1]) * (1 - self.epsilon) / self.q.shape[1]
            p[self.optimal_action_map[s]] += self.epsilon
            return np.random.choice(np.arange(self.q.shape[1]), p=p)

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


def sarsa(env, discount, step_size=0.1, policy_epsilon=0.1, tolerance=0.0001):
    """SARSA for estimating Q*
    
    Args:
        env (PuddleWorldEnv):
        discount (float):
        step_size (float):
        policy_epsilon (float):
        tolerance (float):
    
    Returns:
        (numpy array): |S|x|A| Optimal Q* matrix
    """
    q = np.random.randn(env.num_states, env.num_actions)

    total_steps = 1
    for episode_iter in it.count():

        # Reset the environment
        s1 = env.reset()
        policy = EpsilonGreedyPolicy(q, policy_epsilon / total_steps)
        a1 = policy.act(s1)

        # Roll out an episode
        for step_iter in it.count():
            total_steps += 1
            s2, r1, done, _ = env.step(a1)
            a2 = policy.act(s2)
            delta = step_size * (r1 + discount * q[s2, a2] - q[s1, a1])
            q[s1, a1] += delta

            # Update our plicy
            policy = EpsilonGreedyPolicy(q, policy_epsilon / total_steps)

            s1 = s2
            a1 = a2

            # If we finished the episode, start a new one
            if done:
                break

            # If the Q-function has converged, return
            if np.abs(delta) < tolerance:
                return q


def demo():
    """Demonstrate this module"""

    import matplotlib.pyplot as plt
    from puddle_world.envs import CanonicalPuddleWorldEnv

    # Find the Optimal Q, V functions for the Canonical Puddle World Environment
    env = CanonicalPuddleWorldEnv(mode="dry")
    print(env._ascii())

    q_star = sarsa(env, 0.9, tolerance=1e-5)
    v_star = np.max(q_star, axis=1)

    # Plot Q*
    ax, figs = plt.subplots(1, 4, sharey=True, figsize=(9, 3))
    for ai, a in enumerate(env.actions.keys()):
        plt.sca(figs[ai])
        plt.imshow(q_star[:, ai].reshape((5, 5)))
        plt.xlabel("Action {}".format(a))
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Plot V*
    plt.figure()
    plt.imshow(v_star.reshape((5, 5)))
    plt.colorbar()
    plt.show()

    print("Done")


if __name__ == "__main__":
    demo()
