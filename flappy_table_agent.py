import random
import numpy as np

reward_structures = {
    "basicFlappyAgent": {"positive": 1.0, "tick": 0.0, "loss": -5.0},
    "improvedFlappyAgent": {"positive": 2.0, "tick": 0.5, "loss": -5.0},
    "game": {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
}

def normalize_y(y):
    return int(round(max(y, 0)/(512/14)))


def normalize_x(x):
    return int(round(max(min(x, 288), 0)/(288/14)))


def normalize_player_vel(player_vel):
    return int(round((player_vel+8) / (18/14)))

def improved_map_state(state):
    player_vel = normalize_player_vel(state["player_vel"])
    x_diff = normalize_x(state["next_pipe_dist_to_player"])
    y_diff = normalize_y(state["next_pipe_bottom_y"] - state["player_y"])
    return player_vel*15 + x_diff*(15**2) + y_diff*(15**3)

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        return

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1)

class FlappyQAgent(FlappyAgent):
    def __init__(self, reward_values=reward_structures["basicFlappyAgent"], alpha=0.1, epsilon=0.1, gamma=1):
        self.rewards = reward_values
        self.learning_rate = alpha
        self.epsilon = epsilon
        self.discount_factor = gamma
        self.q_table = np.zeros((50625, 2))

    def reward_values(self):
        return self.rewards

    def observe(self, s1, a, r, s2, end):
        first_state_index = improved_map_state(s1)
        second_state_index = improved_map_state(s2)
        if end:
            self.q_table[second_state_index] = [r, r]
        old_value = self.q_table[first_state_index][a]
        self.q_table[first_state_index][a] = (1-self.learning_rate)*old_value + self.learning_rate*(
            r + self.discount_factor*max(self.q_table[second_state_index]))

    def training_policy(self, state):
        # print("state: %s" % state)
        p = random.random()
        if p < self.epsilon:
            return random.randint(0, 1)
        else:
            stateIndex = improved_map_state(state)
            return np.argmax(self.q_table[stateIndex])

    def policy(self, state):
        # print("state: %s" % state)
        stateIndex = improved_map_state(state)
        return np.argmax(self.q_table[stateIndex])