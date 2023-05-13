import random


class random_agent:
    def __init__(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [force, angle]


class random_agent_2:
    def __init__(self):
        self.force_range = [0, 1]
        self.angle_range = [0, 1]

    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [force, angle]


class static_agent:
    def __init__(self):
        pass

    def act(self, obs):
        force = 0.0
        angle = 0.0

        return [force, angle]

