import random
import time
from typing import Dict

import os
import sys

from environments.olympics_engine.generator import create_scenario
from environments.olympics_engine.scenario import Running_competition, table_hockey, football, wrestling, \
    curling_competition, billiard_joint

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

import gymnasium as gym
import dm_env
from dm_env import specs

# -------------------------------
# T1 0 1 2 3 4 T2
# -------------------------------
import numpy as np
from gymnasium.spaces import Discrete, Box
from ray.rllib.env import EnvContext

from environments.olympics_engine.AI_olympics import AI_Olympics
from environments.olympics_engine.agent import random_agent, random_agent_2, static_agent


def make_game_pool(game_mode):
    max_step = 400
    running_Gamemap = create_scenario("running-competition")
    running_game = Running_competition(running_Gamemap, vis=200, vis_clear=5, agent1_color='light red',
                                            agent2_color='blue')

    tablehockey_game = table_hockey(create_scenario("table-hockey"))
    football_game = football(create_scenario('football'))
    wrestling_game = wrestling(create_scenario('wrestling'))
    curling_game = curling_competition(create_scenario('curling-IJACA-competition'))
    billiard_game = billiard_joint(create_scenario("billiard-joint"))

    running_game.max_step = max_step
    tablehockey_game.max_step = max_step
    football_game.max_step = max_step
    wrestling_game.max_step = max_step

    # random, running, table-hockey, football, wrestling, curling, billiard
    if game_mode == "running":
        game_pool = [{"name": 'running-competition', 'game': running_game}]
    elif game_mode == "table-hockey":
        game_pool = [{"name": 'table-hockey', "game": tablehockey_game}]
    elif game_mode == "football":
        game_pool = [{"name": 'football', "game": football_game}]
    elif game_mode == "wrestling":
        game_pool = [{"name": 'wrestling', "game": wrestling_game}]
    elif game_mode == "curling":
        game_pool = [{"name": "curling", "game": curling_game}]
    elif game_mode == "billiard":
        game_pool = [{"name": "billiard", "game": billiard_game}]
    else:
        game_pool = [{"name": 'running-competition', 'game': running_game},
                          {"name": 'table-hockey', "game": tablehockey_game},
                          {"name": 'football', "game": football_game},
                          {"name": 'wrestling', "game": wrestling_game},
                          {"name": "curling", "game": curling_game},
                          {"name": "billiard", "game": billiard_game}]

    return game_pool


class AiOlympicGym(gym.Env):
    def __init__(
        self,
        env_config: EnvContext
    ):
        self.__version__ = "0.1"
        self.num_steps = None
        self.game_count = 0
        self.shuffled_game = None

        self.entire_obs = None
        self.entire_reward = None

        self.our_team_idx = env_config["our_team_idx"]   # 0 or 1
        self.game = AI_Olympics(random_selection=True, minimap=False)
        self.initial_game_pool = self.game.game_pool

        self.observation_space = Box(low=-np.Inf, high=np.Inf, shape=(40, 40, 1))
        self.action_space = Box(low=0, high=1, shape=(2,))

        if env_config["opponent_type"] == "random":
            self.opponent_agent = random_agent()
        elif env_config["opponent_type"] == "static":
            self.opponent_agent = static_agent()
        else:
            raise ValueError()
        self.game_mode = env_config["game_mode"]

    def reset(self, **kwargs) -> dm_env.TimeStep:
        self.num_steps = 0

        if self.game_count == 0:
            self.game.game_pool = make_game_pool(self.game_mode)
            self.entire_obs = self.game.reset()
            self.shuffled_game = self.game.selected_game_idx_pool

            our_obs = np.expand_dims(self.entire_obs[self.our_team_idx]['agent_obs'], axis=0)
            info = {
                "our_info": self.entire_obs[self.our_team_idx]["info"] if "info" in self.entire_obs[self.our_team_idx] else None,
                "our_id": self.entire_obs[self.our_team_idx]["id"],
                "our_game_mode": self.entire_obs[self.our_team_idx]["game_mode"],
                "our_energy": self.entire_obs[self.our_team_idx]["energy"]
            }
        else:
            self.game.game_pool.pop(self.shuffled_game[0])  # remove the previous game

            self.entire_obs = self.game.reset()
            self.shuffled_game = self.game.selected_game_idx_pool
            our_obs = np.expand_dims(self.entire_obs[self.our_team_idx]['agent_obs'], axis=0)
            info = {
                "our_info": self.entire_obs[self.our_team_idx]["info"] if "info" in self.entire_obs[
                    self.our_team_idx] else None,
                "our_id": self.entire_obs[self.our_team_idx]["id"],
                "our_game_mode": self.entire_obs[self.our_team_idx]["game_mode"],
                "our_energy": self.entire_obs[self.our_team_idx]["energy"]
            }

        if our_obs.dtype != 'float32':
            our_obs = our_obs.astype(np.float32)

        # return our_obs, info
        return dm_env.restart(observation=our_obs)

    def step(self, our_action) -> dm_env.TimeStep:
        self.num_steps += 1

        opponent_action = self.opponent_agent.act(self.entire_obs[1 - self.our_team_idx]['agent_obs'])

        # our_action, opponent_action = self.action_scale(our_action), self.action_scale(opponent_action)
        our_action = self.olympic_action_scale(our_action)

        if self.our_team_idx == 0:
            action = [our_action, opponent_action]
        elif self.our_team_idx == 1:
            action = [opponent_action, our_action]
        else:
            raise ValueError()

        self.entire_obs, self.entire_reward, terminated, game_info = self.game.step(action)

        our_obs = np.expand_dims(self.entire_obs[self.our_team_idx]['agent_obs'], 0)
        if our_obs.dtype != 'float32':
            our_obs = our_obs.astype(np.float32)

        our_reward = self.entire_reward[self.our_team_idx]

        if game_info["reward"][self.our_team_idx] == 1.0:
            our_reward = 1.0

        # Change terminated reward
        # Our team is win reward = 100 -> 1
        if terminated and self.entire_reward[self.our_team_idx] == 100.0:
            our_reward = 1.0
            our_reward += game_info["reward"][self.our_team_idx]

        # reset original ai olympic
        if terminated:
            self.game_count = 0

        # step terminated per game
        if self.entire_obs[self.our_team_idx]["game_mode"] == "NEW GAME":
            self.game_count += 1
            terminated = True

        info = {
            "our_info": self.entire_obs[self.our_team_idx]["info"] if "info" in self.entire_obs[self.our_team_idx] else None,
            "our_id": self.entire_obs[self.our_team_idx]["id"],
            "our_game_mode": self.entire_obs[self.our_team_idx]["game_mode"],
            "our_energy": self.entire_obs[self.our_team_idx]["energy"],
            "game_info": game_info
        }
        truncated = False

        if our_obs.shape != (1, 40, 40):
            our_obs = np.zeros(shape=(1, 40, 40), dtype=np.float32)

        if terminated:
            return dm_env.termination(reward=our_reward, observation=our_obs)
        else:
            return dm_env.transition(reward=our_reward, observation=our_obs)
        # return our_obs, our_reward, terminated, truncated, info

    def render(self):
        self.game.render()

    def olympic_action_scale(self, action):
        # self.force_range = [-100, 200]
        # self.angle_range = [-30, 30]
        # action scale: (action) * (max_action - min_action) + min_action

        scaled_action = []

        if action[0] < 0:
            action[0] = 0.0
        if action[1] < 0:
            action[1] = 0.0

        scaled_action.append(action[0] * (200 + 100) - 100)
        scaled_action.append(action[1] * (30 + 30) - 30)

        return scaled_action

    def observation_spec(self) -> specs.BoundedArray:
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=(1, 40, 40),
            dtype=np.float32,
            name="observation",
            minimum=np.inf,
            maximum=np.inf,
        )

    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            name="action",
            minimum=0,
            maximum=1,
        )

    # def _observation(self) -> np.ndarray:
    #     self._board.fill(0.)
    #     self._board[self._ball_y, self._ball_x] = 1.
    #     self._board[self._paddle_y, self._paddle_x] = 1.
    #
    #     return self._board.copy()


def main():
    our_agent = random_agent_2()

    env_config = {
        "our_team_idx": 0,
        "opponent_type": "static",   # "random"
        "game_mode": "random", # " running, table-hockey, football, wrestling, curling, billiard
    }

    env = AiOlympicGym(env_config)

    for ep in range(12):
        print("\n############ EPISODE: {0} ############".format(ep+1))
        obs, info = env.reset()
        print("reset")
        print("info: {0}".format(info))
        env.render()

        done = False
        total_steps = 0
        while not done:
            total_steps += 1
            our_action = our_agent.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(our_action)
            print("[Step: {0}] obs: {1}, action: [{2:5.3f}:{3:5.3f}], next_obs: {4}, reward: {5}, terminated: {6}, "
                  "our_info: {7}, our_id: {8}, our_game_mode: {9}, our_energy: {10:7.2f}, game: {11}, reward: {12}".format(
                total_steps, obs.shape, our_action[0], our_action[1], next_obs.shape, reward, terminated,
                info['our_info'], info['our_id'], info['our_game_mode'], info['our_energy'], info['game_info']['game'],
                info['game_info']['reward']
            ))
            env.render()

            assert next_obs.shape == (40, 40, 1), \
                "[Ep: {0}, Step: {1}] obs: {2}, action: [{3:5.3f}:{4:5.3f}], next_obs: {5}, reward: {6}, terminated: {7}, " \
                "our_info: {8}, our_id: {9}, our_game_mode: {10}, our_energy: {11:7.2f}, game: {12}, reward: {13}".format(
                    ep, total_steps, obs.shape, our_action[0], our_action[1], next_obs.shape, reward, terminated,
                    info['our_info'], info['our_id'], info['our_game_mode'], info['our_energy'], info['game_info']['game'],
                    info['game_info']['reward']
                )

            obs = next_obs
            done = terminated or truncated

            # time.sleep(0.1)


if __name__ == "__main__":
    main()