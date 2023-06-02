from environments.olympics_engine.scenario import Running_competition, table_hockey, football, wrestling, curling_competition, billiard_joint
import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
from environments.olympics_engine.generator import create_scenario

import random


class AI_Olympics:
    def __init__(self, random_selection, minimap):

        self.random_selection = True
        self.minimap_mode = minimap

        self.max_step = 400
        self.vis = 200
        self.vis_clear = 5

        running_Gamemap = create_scenario("running-competition")
        self.running_game = Running_competition(running_Gamemap, vis = 200, vis_clear=5, agent1_color = 'light red', agent2_color='blue')

        self.tablehockey_game = table_hockey(create_scenario("table-hockey"))
        self.football_game = football(create_scenario('football'))
        self.wrestling_game = wrestling(create_scenario('wrestling'))
        self.curling_game = curling_competition(create_scenario('curling-IJACA-competition'))
        self.billiard_game = billiard_joint(create_scenario("billiard-joint"))

        self.running_game.max_step = self.max_step
        self.tablehockey_game.max_step = self.max_step
        self.football_game.max_step = self.max_step
        self.wrestling_game.max_step = self.max_step
        # self.curling_game.max_step =

        self.game_pool = [{"name": 'running-competition', 'game': self.running_game},
                          {"name": 'table-hockey', "game": self.tablehockey_game},
                          {"name": 'football', "game": self.football_game},
                          {"name": 'wrestling', "game": self.wrestling_game},
                          {"name": "curling", "game": self.curling_game},
                          {"name": "billiard", "game": self.billiard_game}]
        self.view_setting = self.running_game.view_setting
        self.episode_steps = 0

    def reset(self):
        self.done = False
        selected_game_idx_pool = list(range(len(self.game_pool)))
        if self.random_selection:
            random.shuffle(selected_game_idx_pool)            #random game playing sequence

        self.selected_game_idx_pool = selected_game_idx_pool                           #fix game playing sequence
        self.current_game_count = 0
        selected_game_idx = self.selected_game_idx_pool[self.current_game_count]

        self.episode_steps = 0
        # print(f'[###RESET###] Playing {self.game_pool[selected_game_idx]["name"]}')

        # if self.game_pool[selected_game_idx]['name'] == 'running-competition':
        #     self.game_pool[selected_game_idx]['game'] = \
        #         Running_competition.reset_map(meta_map= self.running_game.meta_map,map_id=None, vis=200, vis_clear=5,
        #                                       agent1_color = 'light red', agent2_color = 'blue')     #random sample a map
        #     self.game_pool[selected_game_idx]['game'].max_step = self.max_step

        self.current_game = self.game_pool[selected_game_idx]['game']
        self.game_score = [0, 0]

        init_obs = self.current_game.reset()
        if self.current_game.game_name == 'running-competition':
            init_obs = [{'agent_obs': init_obs[i], 'id': f'team_{i}'} for i in [0, 1]]

        for i in init_obs:
            i['game_mode'] = 'NEW GAME'

        for i,j in enumerate(init_obs):
            if 'curling' in self.current_game.game_name:
                j['energy'] = 1000
            else:
                j['energy'] = self.current_game.agent_list[i].energy

        return init_obs

    def get_state(self):
        state = []
        # wall, cross, arc
        state.append([[0.0 for _ in range(7)] for w in range(12)])
        state.append([[0.0 for _ in range(6)] for w in range(13)])
        state.append([[0.0 for _ in range(7)] for w in range(4)])

        agent_pos_max = 1100
        wall_pos_max = 1100
        wall_length_max = 1100
        wall_width_max = 2
        cross_pos_max = 1100
        cross_length_max = 1100
        cross_width_max = 5
        arc_pos_max = 1100
        arc_radian_max = 180

        n_wall = 0
        n_cross = 0
        n_arc = 0

        for object_idx in range(len(self.map["objects"])):
            object = self.map["objects"][object_idx]

            wall = []
            if object.type == 'wall':
                wall_pos = [y / wall_pos_max for a in object.init_pos for y in a]
                assert max(wall_pos) <= 1.0
                wall += wall_pos
                if object.ball_can_pass:
                    wall.append(1.0)
                else:
                    wall.append(0.0)
                wall_length = object.length / wall_length_max
                assert wall_length <= 1.0
                wall_width = object.width / wall_width_max
                assert wall_length <= 1.0
                wall.append(wall_length)
                wall.append(wall_width)
                state[0][n_wall] = wall
                n_wall += 1

            cross = []
            if object.type == 'cross':
                cross_pos = [y / cross_pos_max for a in object.init_pos for y in a]
                assert max(cross_pos) <= 1.0
                cross += cross_pos
                cross_length = object.length / cross_length_max
                assert cross_length <= 1.0
                cross_width = object.width / cross_width_max
                assert cross_width <= 1.0
                cross.append(cross_length)
                cross.append(cross_width)
                state[1][n_cross] = cross
                n_cross += 1

            arc = []
            if object.type == 'arc':
                arc_pos = [p / arc_pos_max for p in object.init_pos]
                assert max(arc_pos) <= 1.0
                arc += arc_pos
                if object.ball_can_pass:
                    arc.append(1.0)
                else:
                    arc.append(0.0)
                arc_start_radian = object.start_radian / arc_radian_max
                assert arc_start_radian <= 1.0
                arc_end_radian = object.end_radian / arc_radian_max
                assert arc_end_radian <= 1.0
                arc.append(arc_start_radian)
                arc.append(arc_end_radian)
                state[2][n_arc] = arc
                n_arc += 1
        state_flattened = [y for a in state for x in a for y in x]

        agent = [0.0 for _ in range(6)]
        agents_pos = [y / agent_pos_max for a in self.agent_pos for y in a]
        assert max(agents_pos) <= 1.0
        for i, a in enumerate(agents_pos):
            agent[i] = a
        full_state = agents_pos + state_flattened

        return full_state

    def step(self, action_list):
        self.episode_steps += 1
        obs, reward, done, info = self.current_game.step(action_list)

        if self.current_game.game_name == 'running-competition':
            obs = [{'agent_obs': obs[i], 'id': f'team_{i}'} for i in [0,1]]
        for i in obs:
            i['game_mode'] = ''

        for i, j in enumerate(obs):
            if 'curling' in self.current_game.game_name:
                j['energy'] = 1000
            elif 'billiard' in self.current_game.game_name:
                j['energy'] = self.current_game.agent_energy[i]
            else:
                j['energy'] = self.current_game.agent_list[i].energy

        if done:
            winner = self.current_game.check_win()
            if winner != '-1':
                self.game_score[int(winner)] += 1

            if self.current_game_count == len(self.game_pool) - 1:
                self.done = True
            else:
                # self.current_game_idx += 1
                self.current_game_count += 1
                self.current_game_idx = self.selected_game_idx_pool[self.current_game_count]

                self.current_game = self.game_pool[self.current_game_idx]['game']
                #print(f'[Step: {self.episode_steps}] Playing {self.game_pool[self.current_game_idx]["name"]}')
                obs = self.current_game.reset()

                if self.current_game.game_name == 'running-competition':
                    obs = [{'agent_obs': obs[i], 'id': f'team_{i}'} for i in [0, 1]]

                for i in obs:
                    i['game_mode'] = 'NEW GAME'
                for i,j in enumerate(obs):
                    if 'curling' in self.current_game.game_name:
                        j['energy'] = 1000
                    else:
                        j['energy'] = self.current_game.agent_list[i].energy

        game_info = {
            "game": self.current_game.game_name,
            "reward": reward
        }
        if self.done:
            #print('[DONE] game score = {0}'.format(self.game_score), end=": ")
            if self.game_score[0] > self.game_score[1]:
                self.final_reward = [100, 0]
                #print('Results: team 0 win!')
            elif self.game_score[1] > self.game_score[0]:
                self.final_reward = [0, 100]
                #print('Results: team 1 win!')
            else:
                self.final_reward = [0, 0]
                #print('Results: Draw!')

            return obs, self.final_reward, self.done, game_info
        else:
            return obs, reward, self.done, game_info

    def is_terminal(self):
        return self.done

    def __getattr__(self, item):
        return getattr(self.current_game, item)


    def render(self):
        self.current_game.render()



