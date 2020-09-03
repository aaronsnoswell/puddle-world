import gym
import pyglet
import interface
import numpy as np
import itertools as it
import matplotlib.pyplot as plt

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control import rendering

from explicit_env import IExplicitEnv, ExplicitEnvGetters
from explicit_env.envs.utils import compute_parents_children


class ExplicitPuddleWorldEnv(
    gym.Env, interface.implements(IExplicitEnv), ExplicitEnvGetters
):

    # Features of the environment
    FEATURES = {"dry": 0, "wet": 1, "goal": 2}

    # Actions the agent can take
    ACTION_MAP = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3,
    }

    ACTION_SYMBOLS = {
        "up": "↑",
        "down": "↓",
        "left": "←",
        "right": "→",
    }

    # Coordinate system is origin at top left, +Y down, +X right
    ACTION_VECTORS = {
        0: np.array([-1, 0]),
        1: np.array([1, 0]),
        2: np.array([0, -1]),
        3: np.array([0, 1]),
    }

    # Rewards values
    REWARD_VALUES = {"very_bad": -10, "bad": -1, "meh": 0}

    # Different reward modes
    REWARD_MODES = {
        "wet": [REWARD_VALUES["very_bad"], REWARD_VALUES["bad"], REWARD_VALUES["meh"]],
        "dry": [REWARD_VALUES["bad"], REWARD_VALUES["very_bad"], REWARD_VALUES["meh"]],
        "any": [REWARD_VALUES["bad"], REWARD_VALUES["bad"], REWARD_VALUES["meh"]],
    }

    # Probability of a non-goal state being wet
    P_WET = 5.0 / 24.0

    # Probability of a non-goal state being a start state
    P_START = 11.0 / 24.0

    # Gym Env properties
    metadata = {"render.modes": ["human", "rgb_array", "ascii"]}
    reward_range = (min(REWARD_VALUES.values()), max(REWARD_VALUES.values()))

    def __init__(self, width, height, *, mode="dry", wind=0.2, seed=None):
        """C-tor
        
        Args:
            width (int): Width of PuddleWorld
            height (int): Height of PuddleWorld
            
            mode (str): Reward mode to use, options are 'wet', 'dry', and 'any'
            wind (float): Wind (random action) probability
            seed (int): Random seed to use
        """

        assert mode in self.REWARD_MODES.keys()
        self._mode = mode

        self.seed(seed)

        self._width = width
        self._height = height
        self.observation_space = spaces.Discrete(self._width * self._height)
        self.action_space = spaces.Discrete(len(self.ACTION_MAP))

        self._wind = wind

        # Build the feature matrix
        # 1. Select random goal
        goal_state = np.random.choice(np.arange(self.observation_space.n))
        self._feature_matrix = np.zeros((self._height, self._width), dtype=int)
        self._feature_matrix.flat[goal_state] = self.FEATURES["goal"]

        while True:

            # 2. Non-goal states may be wet/dry
            for s, (y, x) in enumerate(
                it.product(range(self._height), range(self._width))
            ):
                if s == goal_state:
                    continue
                self._feature_matrix[y, x] = np.random.rand() <= self.P_WET

            # 3. Non-goal states may be start states
            self._start_states = []
            for s in range(self.observation_space.n):
                if s == goal_state:
                    continue
                if np.random.rand() <= self.P_START:
                    self._start_states.append(s)
            self._start_states = np.array(self._start_states)

            # Check that we have at least one wet and one dry
            # and that we have at least one start state
            if (
                self.FEATURES["wet"] in self._feature_matrix.flat
                and self.FEATURES["dry"] in self._feature_matrix.flat
                and len(self._start_states) > 0
            ):
                break

        # Prepare IExplicitEnv items
        self._states = np.arange(self.observation_space.n, dtype=int)
        self._actions = np.arange(self.action_space.n, dtype=int)

        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[self._start_states] = 1.0
        self._p0s /= np.sum(self._p0s)

        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[goal_state] = 1.0

        # Compute s, a, s' transition matrix
        self._t_mat = self._build_transition_matrix()

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        self._gamma = 0.99

        # Build linear state reward vector
        self._state_rewards = np.array(
            [
                self.REWARD_MODES[self._mode][self._feature_matrix.flat[s]]
                for s in self._states
            ]
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None

        self.state = self.reset()

    def seed(self, seed=None):
        """Seed the environment"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _build_transition_matrix(self):
        """Assemble the transition matrix from self._feature_matrix"""
        # Compute s, a, s' transition matrix
        transition_matrix = np.zeros(
            (len(self.states), len(self.actions), len(self.states))
        )
        for s1, a in it.product(self.states, self.actions):
            # Convert states to coords, action to vector
            yx1 = np.array(self._s2yx(s1))
            av = ExplicitPuddleWorldEnv.ACTION_VECTORS[a]

            # If moving out of bounds, return to current state
            if self._oob(yx1 + av):
                transition_matrix[s1, a, s1] = 1.0
            else:
                target_state = self._yx2s(yx1 + av)
                alternate_states = self._nei(s1)
                alternate_states.remove(target_state)

                # Wind might move us to an alternate state
                transition_matrix[s1, a, alternate_states] = self._wind / len(
                    alternate_states
                )

                # Target state gets non-wind probability
                transition_matrix[s1, a, target_state] = 1.0 - self._wind

        # Ensure that goal state(s) are terminal
        goal_states = np.where(
            self._feature_matrix.flatten() == ExplicitPuddleWorldEnv.FEATURES["goal"]
        )
        transition_matrix[goal_states, :, :] = 0.0

        return transition_matrix

    def reset(self):
        """Reset the environment"""
        self.state = np.random.choice(self.states, p=self.p0s)
        return self.observe()

    def step(self, action):
        """Step the environment"""

        # Verify action
        assert self.action_space.contains(action), "Invalid action: {}".format(action)

        # Apply action
        self.state = np.random.choice(
            self.states, p=self._t_mat[self.state, action, :].flatten(),
        )

        return (
            self.observe(self.state),
            self.reward(self.state),
            self.done(self.state),
            {},
        )

    def _s2yx(self, state):
        """Convert state to (y, x) coordinates"""
        assert self.observation_space.contains(state)
        y = state // self._width
        x = state - y * self._width
        return (y, x)

    def _yx2s(self, yx):
        """Convert y, x tuple to state"""
        y, x = yx
        assert 0 <= y < self._height
        assert 0 <= x < self._width
        return y * self._width + x

    def _oob(self, yx):
        """Check if a y, x coordinate is 'out of bounds'"""
        try:
            return not self.observation_space.contains(self._yx2s(yx))
        except AssertionError:
            return True

    def _nei(self, state=None):
        """Get neighbours of a state"""
        if state is None:
            state = self.state

        y, x = self._s2yx(state)
        neighbours = []
        if y > 0:
            neighbours.append(self._yx2s((y - 1, x)))
        if y < self._height - 1:
            neighbours.append(self._yx2s((y + 1, x)))
        if x > 0:
            neighbours.append(self._yx2s((y, x - 1)))
        if x < self._width - 1:
            neighbours.append(self._yx2s((y, x + 1)))

        return neighbours

    def observe(self, state=None):
        """Get an observation for a state"""
        if state is None:
            state = self.state
        return int(state)

    def reward(self, state):
        """Compute reward given state"""
        if state is None:
            state = self.state
        return self._state_rewards[state]

    def done(self, state):
        """Test if a episode is complete"""
        if state is None:
            state = self.state
        return self._terminal_state_mask[state]

    def render(self, mode="human"):
        """Render the environment"""
        assert mode in self.metadata["render.modes"]

        if mode == "ascii":
            return self._ascii()
        else:
            raise NotImplementedError

    def _ascii(self):
        """Get an ascii string representation of the environment"""
        str_repr = "+" + "-" * self._width + "+\n"
        for row in range(self._height):
            str_repr += "|"
            for col in range(self._width):
                state = self._yx2s((row, col))
                state_feature = self._feature_matrix.flatten()[state]
                if state == self.state:
                    str_repr += "@"
                elif state_feature == self.FEATURES["dry"]:
                    str_repr += " "
                elif state_feature == self.FEATURES["wet"]:
                    str_repr += "#"
                else:
                    str_repr += "G"
            str_repr += "|\n"
        str_repr += "+" + "-" * self._width + "+"
        return str_repr


class CanonicalPuddleWorldEnv(ExplicitPuddleWorldEnv):
    """The canonical puddle world environment"""

    def __init__(self, **kwargs):

        super().__init__(5, 5, **kwargs)

        # Specify the canonical feature matrix
        self._feature_matrix = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 2],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        # Specify start states
        self._start_states = np.array(
            [0, 1, 2, 5, 6, 10, 15, 16, 20, 21, 22], dtype=np.int64
        )

        goal_state = 14

        # Prepare IExplicitEnv items
        self._states = np.arange(self.observation_space.n, dtype=int)
        self._actions = np.arange(self.action_space.n, dtype=int)

        self._p0s = np.zeros(self.observation_space.n, dtype=float)
        self._p0s[self._start_states] = 1.0
        self._p0s /= np.sum(self._p0s)

        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[goal_state] = 1.0

        # Compute s, a, s' transition matrix
        self._t_mat = self._build_transition_matrix()

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        self._gamma = 0.99

        # Build linear state reward vector
        self._state_rewards = np.array(
            [
                self.REWARD_MODES[self._mode][self._feature_matrix.flat[s]]
                for s in self._states
            ],
            dtype=float,
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None

        self.state = self.reset()

        self._p0s = np.zeros(self.observation_space.n)
        self._p0s[self._start_states] = 1.0
        self._p0s /= np.sum(self._p0s)

        self._terminal_state_mask = np.zeros(self.observation_space.n)
        self._terminal_state_mask[goal_state] = 1.0

        # Compute s, a, s' transition matrix
        self._t_mat = self._build_transition_matrix()

        self._parents, self._children = compute_parents_children(
            self._t_mat, self._terminal_state_mask
        )

        self._gamma = 0.99

        # Build linear state reward vector
        self._state_rewards = np.array(
            [
                self.REWARD_MODES[self._mode][self._feature_matrix.flat[s]]
                for s in self._states
            ]
        )
        self._state_action_rewards = None
        self._state_action_state_rewards = None

        self.state = self.reset()


def demo():
    """Demonstrate this task"""

    env = CanonicalPuddleWorldEnv(mode="dry")

    # Check the environment is compliant
    # from stable_baselines.common.env_checker import check_env
    # check_env(env)

    print(env._ascii())

    # Train a PPO2 agent
    # from stable_baselines.common.policies import MlpPolicy
    # from stable_baselines import PPO2
    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=int(60e4))

    # Can now evaluate performance...

    print("Done")


if __name__ == "__main__":
    demo()
