import gym

class GymGame:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.reset()

    def reset(self):
        self.current_state = self.env.reset()
        self.score = 0

    def playStep(self, action):
        self.env.render()
        results = self.env.step(action)
        self.score += 1
        state = results[0]
        self._set_state(state)
        reward = results[1]
        done = results[2]
        return state, reward, done, self.score

    def _set_state(self, state):
        self.current_state = state

    def get_state(self):
        return self.current_state
