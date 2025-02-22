# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35
        self.repr_dim = 5408

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class AutoEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3

        self.encoder_hidden_layer1 = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            nn.ReLU()
        )
        self.encoder_hidden_layer2 = nn.Sequential(
            nn.Conv2d(32, 16, 2, stride=2),
            nn.ReLU()
        )
        self.encoder_hidden_layer3 = nn.Sequential(
            nn.Conv2d(16, 16, 2, stride=2),
            nn.ReLU()
        )
        self.encoder_output_layer = nn.Sequential(
            nn.Conv2d(16, 8, 1, stride=1),
            nn.ReLU()
        )
        self.decoder_hidden_layer1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 1, stride=1),
            nn.ReLU()
        )
        self.decoder_hidden_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU()
        )
        self.decoder_hidden_layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 2, stride=2),
            nn.ReLU()
        )
        self.decoder_output_layer = nn.Sequential(
            nn.ConvTranspose2d(32, obs_shape[0], 2, stride=2),
            nn.Sigmoid()
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        activation = self.encoder_hidden_layer1(obs)
        activation = self.encoder_hidden_layer2(activation)
        activation = self.encoder_hidden_layer3(activation)
        encode = self.encoder_output_layer(activation)
        activation = self.decoder_hidden_layer1(encode)
        activation = self.decoder_hidden_layer2(activation)
        activation = self.decoder_hidden_layer3(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)

        return reconstructed


class StateEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3

        self.encoder_hidden_layer1 = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            nn.ReLU()
        )
        self.encoder_hidden_layer2 = nn.Sequential(
            nn.Conv2d(32, 16, 2, stride=2),
            nn.ReLU()
        )
        self.encoder_hidden_layer3 = nn.Sequential(
            nn.Conv2d(16, 16, 2, stride=2),
            nn.ReLU()
        )
        self.encoder_output_layer = nn.Sequential(
            nn.Conv2d(16, 8, 1, stride=1),
            nn.ReLU()
        )
        self.decoder_hidden_layer1 = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 1, stride=1),
            nn.ReLU()
        )
        self.decoder_hidden_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),
            nn.ReLU()
        )
        self.decoder_hidden_layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 2, stride=2),
            nn.ReLU()
        )
        self.decoder_output_layer = nn.Sequential(
            nn.ConvTranspose2d(32, obs_shape[0], 2, stride=2),
            nn.Sigmoid()
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        activation = self.encoder_hidden_layer1(obs)
        activation = self.encoder_hidden_layer2(activation)
        activation = self.encoder_hidden_layer3(activation)
        encode = self.encoder_output_layer(activation)
        activation = self.decoder_hidden_layer1(encode)
        activation = self.decoder_hidden_layer2(activation)
        activation = self.decoder_hidden_layer3(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)

        return reconstructed


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.auto_encoder = AutoEncoder(obs_shape).to(device)
        self.state_encoder = StateEncoder([1, 40, 40]).to(device)

        # MSE loss
        self.mse_loss = nn.MSELoss(size_average=None, reduce=None, reduction='none')

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.auto_encoder_opt = torch.optim.Adam(self.auto_encoder.parameters(), lr=lr)
        self.state_encoder_opt = torch.optim.Adam(self.state_encoder.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.auto_encoder.train(training)
        self.state_encoder.train(training)

    def act(self, obs, step, eval_mode):
        obs_len = obs.shape[0]
        state = obs[-1:]

        obs = obs[:obs_len - 1]
        obs = torch.as_tensor(obs, device=self.device)
        state_encoding = self.state_encoder(obs)
        # print("obs_shape: ", obs.shape)
        # print("state encoder: ", state_encoding.shape)
        if state_encoding.shape[0] > 1:
            obs = torch.cat([obs, state_encoding[:-1]], dim=0)
        else:
            obs = torch.cat([obs, state_encoding], dim=0)
        # obs = torch.cat([obs, state_encoding[:-1]], dim=0)
        # obs = torch.cat([obs, state_encoding], dim=0)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_auto_encoder(self, next_obs, step):
        metrics = dict()

        bs = next_obs.shape[0]
        obs_len = next_obs.shape[1]
        state = next_obs[:, -1:]
        next_obs = next_obs[:, :obs_len - 1]

        decoded_next_obs = self.auto_encoder(next_obs)
        decoded_next_obs = decoded_next_obs.reshape((bs, -1))
        next_obs = next_obs.reshape((bs, -1)) / 8.0
        loss = self.mse_loss(decoded_next_obs, next_obs).mean(1)
        intrinsic_rewards = loss.clone()
        loss = loss.mean()

        self.auto_encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.auto_encoder_opt.step()

        intrinsic_rewards *= 0.005

        if self.use_tb:
            metrics['intrinsic_rewards'] = intrinsic_rewards.mean().item()
            metrics['auto_encoder_loss'] = loss.item()

        # print("loss: ", loss.item())

        return intrinsic_rewards, metrics

    def update_state_encoder(self, obs, state):
        metrics = dict()

        state_encoding = self.state_encoder(obs)
        if state_encoding.shape[1] > 1:
            state_encoding = state_encoding[:, :-1]
        loss = self.mse_loss(state_encoding, state).mean(1)
        loss = loss.mean()

        self.state_encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.state_encoder_opt.step()

        if self.use_tb:
            metrics['state_encoder_loss'] = loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # # update autoencoder
        # intrinsic_reward, ae_metrics = self.update_auto_encoder(next_obs, step)
        # intrinsic_reward = intrinsic_reward.unsqueeze(1)
        # reward += intrinsic_reward.detach()
        #
        # # update autoencoder metrics
        # metrics.update(ae_metrics)

        # encode
        obs_len = obs.shape[1]
        next_obs_len = next_obs.shape[1]
        state = obs[:, -1:]
        obs = obs[:, :obs_len - 1]

        next_state = next_obs[:, -1:]
        next_obs = next_obs[:, :next_obs_len - 1]

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # global state encoding
        state_encoding = self.state_encoder(obs)
        next_state_encoding = self.state_encoder(next_obs)

        # update state encoder
        metrics.update(self.update_state_encoder(obs, state))

        if state_encoding.shape[1] > 1:
            obs = torch.cat([obs, state_encoding[:, :-1]], dim=1)
        else:
            obs = torch.cat([obs, state_encoding], dim=1)
        obs = self.encoder(obs)
        with torch.no_grad():
            if next_state_encoding.shape[1] > 1:
                next_obs = torch.cat([next_obs, next_state_encoding[:, :-1]], dim=1)
            else:
                next_obs = torch.cat([next_obs, next_state_encoding], dim=1)

            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
