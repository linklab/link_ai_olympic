# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from datetime import datetime

from dm_control.suite.wrappers import action_scale

from environments.ai_olympic_gym import AiOlympicGym

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

# torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        # self.model_save_dir = "/home/yh21h/git/link_ai_olympic/models"
        self.model_save_dir = os.path.dirname(os.path.realpath(__file__)) + "/models"
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        if self.cfg.self_competition:
            self.opponent_agent = make_agent(self.train_env.observation_spec(),
                                    self.train_env.action_spec(),
                                    self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        if self.cfg.task_name == 'ai_olympic':
            env_config = {
                "our_team_idx": self.cfg.our_team_idx,
                "opponent_type": self.cfg.opponent_type,  # "random"
                "game_mode": self.cfg.game_mode,  # " running, table-hockey, football, wrestling, curling, billiard
                "self_competition": self.cfg.self_competition # "True", "False"
            }

            self.train_env = AiOlympicGym(env_config)
            self.eval_env = AiOlympicGym(env_config)

            self.train_env = dmc.ActionDTypeWrapper(self.train_env, np.float32)
            self.train_env = dmc.ActionRepeatWrapper(self.train_env, 1)
            self.train_env = action_scale.Wrapper(self.train_env, minimum=0.0, maximum=+1.0)
            # self.train_env = dmc.FrameStackWrapper(self.train_env, 1, 'pixels')
            self.train_env = dmc.ExtendedTimeStepWrapper(self.train_env)

            self.eval_env = dmc.ActionDTypeWrapper(self.eval_env, np.float32)
            self.eval_env = dmc.ActionRepeatWrapper(self.eval_env, 1)
            self.eval_env = action_scale.Wrapper(self.eval_env, minimum=0.0, maximum=+1.0)
            # self.eval_env = dmc.FrameStackWrapper(self.eval_env, 1, 'pixels')
            self.eval_env = dmc.ExtendedTimeStepWrapper(self.eval_env)
        else:
            self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                      self.cfg.action_repeat, self.cfg.seed)
            self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                     self.cfg.action_repeat, self.cfg.seed)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array(shape=(1,), dtype=np.float32, name='reward'),
                      specs.Array(shape=(1,), dtype=np.float32, name='discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                # if self.cfg.self_competition and self.global_episode != 0:
                #     self.opponent_agent.encoder = torch.load(self.model_save_dir + "/encoder.pth")
                #     self.opponent_agent.actor = torch.load(self.model_save_dir + "/actor.pth")
                with torch.no_grad(), utils.eval_mode(self.agent):
                    # if self.cfg.self_competition:
                    #     action = self.agent.act(time_step.observation,
                    #                             self.global_step,
                    #                             eval_mode=True)
                    #     opponent_action = self.opponent_agent.act(time_step.observation,
                    #                                      self.global_step,
                    #                                      eval_mode=True)
                    # else:
                    action = self.agent.act(time_step.observation,
                                                     self.global_step,
                                                     eval_mode=True)
                # if self.cfg.self_competition:
                #     self.eval_env.set_opponent_action(opponent_action)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            # self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()

        print("state: ", self.train_env.get_state())

        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # if self.cfg.self_competition and self._global_episode % self.cfg.model_save_episode == 0:
            #     if self._global_episode != 0:
            #         self.opponent_agent.encoder = torch.load(self.model_save_dir + "/encoder.pth")
            #         self.opponent_agent.actor = torch.load(self.model_save_dir + "/actor.pth")

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                # if self.cfg.self_competition:
                #     action = self.agent.act(time_step.observation,
                #                             self.global_step,
                #                             eval_mode=True)
                #     opponent_action = self.opponent_agent.act(time_step.observation,
                #                                               self.global_step,
                #                                               eval_mode=True)
                # else:
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=True)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            # if self.cfg.self_competition:
            #     self.train_env.set_opponent_action(opponent_action)
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)

            if self._global_episode % self.cfg.model_save_episode == 0:
                torch.save(self.agent.encoder, self.model_save_dir + "/encoder.pth")
                torch.save(self.agent.actor, self.model_save_dir + "/actor.pth")
                torch.save(self.agent.critic, self.model_save_dir + "/critic.pth")
                torch.save(self.agent.aug, self.model_save_dir + "/aug.pth")

            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()