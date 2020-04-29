import gym
import pyglet
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering


class PuddleWorldEnv(gym.Env):
    """A discrete multi-modal environment
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    # Features of the environment
    features = {"dry": 0, "wet": 1, "goal": 2}

    # Actions the agent can take
    actions = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
    }

    # Coordinate system is origin at top left, +Y down, +X right
    action_vectors = {
        0: np.array([-1, 0]),
        1: np.array([1, 0]),
        2: np.array([0, -1]),
        3: np.array([0, 1]),
    }

    # Different reward modes
    reward_modes = {
        "wet": [-10, -1, 0],
        "dry": [-1, -10, 0],
        "any": [-1, -1, 0],
    }

    @property
    def num_states(self):
        return self.state_space.n

    @property
    def num_features(self):
        return len(self.features)

    @property
    def num_actions(self):
        return len(self.actions)

    def __init__(self, *, mode="wet", wind=0.2, goal_absorbing=True, seed=None):
        """C-tor
        
        Args:
            mode (str): Reward mode to use, options are 'wet', 'dry', and 'any'
            wind (float): Wind (random action) probability
            goal_absorbing (bool): If true, the goal state is absorbing
            seed (int): Random seed to use
        """

        assert mode in self.reward_modes.keys()
        self.mode = mode

        self.seed(seed)

        self.width = 5
        self.height = 5
        self.state_space = spaces.Discrete(self.width * self.height)
        self.observation_space = spaces.Discrete(self.num_features)
        self.action_space = spaces.Discrete(self.num_actions)

        # Populate the world
        self.wind = wind
        self.feature_matrix = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 2],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        self.start_states = [0, 1, 2, 5, 6, 10, 15, 16, 20, 21, 22]

        # Compute s, a, s' transition matrix
        self.transition_matrix = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )
        for s1, a in it.product(range(self.num_states), range(self.num_actions)):
            # Convert states to coords, action to vector
            yx1 = np.array(self.s2yx(s1))
            av = self.action_vectors[a]

            # If moving out of bounds, return to current state
            if self.oob(yx1 + av):
                self.transition_matrix[s1, a, s1] = 1.0
            else:
                target_state = self.yx2s(yx1 + av)
                alternate_states = self.nei(s1)
                alternate_states.remove(target_state)

                # Wind might move us to an alternate state
                self.transition_matrix[s1, a, alternate_states] = self.wind / len(
                    alternate_states
                )

                # Target state gets non-wind probability
                self.transition_matrix[s1, a, target_state] = 1.0 - wind

        if goal_absorbing:
            # Make the goal state(s) absorbing
            goal_states = np.where(
                self.feature_matrix.flatten() == self.features["goal"]
            )
            self.transition_matrix[goal_states, :, :] = 0.0
            for g in goal_states:
                self.transition_matrix[g, :, g] = 1.0

        self.state = self.reset()

    def seed(self, seed=None):
        """Seed the environment"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Reset the environment"""
        self.state = np.random.choice(self.start_states)
        return self.observe()

    def s2yx(self, state):
        """Convert state to (y, x) coordinates"""
        assert self.state_space.contains(state)
        y = state // self.width
        x = state - y * self.width
        return (y, x)

    def yx2s(self, yx):
        """Convert y, x tuple to state"""
        y, x = yx
        assert 0 <= y < self.height
        assert 0 <= x < self.width
        s = y * self.width + x
        return s

    def oob(self, yx):
        """Check if a y, x coordinate is 'out of bounds'"""
        try:
            return not self.state_space.contains(self.yx2s(yx))
        except AssertionError:
            return True

    def observe(self, state=None):
        """Get an observation for a state"""
        if state is None:
            state = self.state
        return state

    def nei(self, state=None):
        """Get neighbours of a state"""
        if state is None:
            state = self.state

        y, x = self.s2yx(state)
        neighbours = []
        if y > 0:
            neighbours.append(self.yx2s((y - 1, x)))
        if y < self.height - 1:
            neighbours.append(self.yx2s((y + 1, x)))
        if x > 0:
            neighbours.append(self.yx2s((y, x - 1)))
        if x < self.width - 1:
            neighbours.append(self.yx2s((y, x + 1)))

        return neighbours

    def step(self, action):
        """Step the environment"""

        # Verify action
        assert self.action_space.contains(action), "Invalid action: {}".format(action)

        # Apply action
        self.state = np.random.choice(
            np.arange(self.num_states),
            p=self.transition_matrix[self.state, action].flatten(),
        )

        # Get reward
        reward = self.reward(self.state)

        # Check if complete
        done = self.done(self.state)

        # Store matadata comment
        meta = {}

        return self.observe(self.state), reward, done, meta

    def reward(self, state):
        """Compute reward given state"""
        if state is None:
            state = self.state
        reward_weights = self.reward_modes[self.mode]
        state_feature = self.feature_matrix.flatten()[state]
        return reward_weights[state_feature]

    def done(self, state):
        """Test if a episode is complete"""
        if state is None:
            state = self.state
        return self.feature_matrix.flatten()[state] == self.features["goal"]

    def render(self, mode="human"):
        """Render the environment"""
        raise NotImplementedError


def demo():
    """Demonstrate this task"""
    env = PuddleWorldEnv()

    o0 = env.reset()


if __name__ == "__main__":
    demo()
