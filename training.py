#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 导入训练基本环境
import argparse
import os
import shutil
from random import random, randint, sample
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# 使用游戏本体代码与Q-network代码创建环境
from core.deep_q_network import DuelingDQN
from core.game import Tetris


def get_args():
    parser = argparse.ArgumentParser(
        """使用 Dueling Double DQN 实现AI自主游玩Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--target_score", type=int, default=0,
                        help="达到该分数后提前结束训练并保存模型，0 表示不限制")
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Size of replay memory pool")
    parser.add_argument("--target_update", type=int, default=500,
                        help="Number of epochs between target network updates")
    parser.add_argument("--lr_decay_step", type=int, default=1000,
                        help="Number of epochs between learning rate decay")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5,
                        help="Learning rate decay factor")
    parser.add_argument("--grad_clip", type=float, default=10,
                        help="Max norm for gradient clipping")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Enable game rendering during training")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    os.makedirs(opt.saved_path, exist_ok=True)
    writer = SummaryWriter(opt.log_path)
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DuelingDQN()
    target_model = DuelingDQN()
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # 学习率衰减：每 lr_decay_step 个 epoch 乘以 lr_decay_gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_decay_step, gamma=opt.lr_decay_gamma)
    # Huber Loss：对异常值更鲁棒，防止梯度爆炸
    criterion = nn.SmoothL1Loss()

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        target_model.cuda()
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    best_score = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :]
        action = next_actions[index]

        reward, done = env.step(action, render=opt.render)

        if torch.cuda.is_available():
            next_state = next_state.cuda()
        replay_memory.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.batch_size:
            continue
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)

        # Double DQN: 主网络选动作，Target Network 评估价值
        target_model.eval()
        with torch.no_grad():
            # 主网络选择下一状态的最优动作
            best_actions = model(next_state_batch).argmax(1, keepdim=True)
            # Target Network 评估该动作的 Q 值
            next_prediction_batch = target_model(next_state_batch).gather(1, best_actions)

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        scheduler.step()

        # 定期同步 target network
        if epoch % opt.target_update == 0:
            target_model.load_state_dict(model.state_dict())

        print("Epoch: {}/{}, Loss: {:.4f}, LR: {:.6f}, Epsilon: {:.4f}, Score: {}, Tetrominoes: {}, Cleared lines: {}, Best: {}".format(
            epoch,
            opt.num_epochs,
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epsilon,
            final_score,
            final_tetrominoes,
            final_cleared_lines,
            best_score))
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)
        writer.add_scalar('Train/Loss', loss.item(), epoch - 1)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch - 1)
        writer.add_scalar('Train/Epsilon', epsilon, epoch - 1)

        # 保存最佳模型
        if final_score > best_score:
            best_score = final_score
            torch.save(model, "{}/tetris_best".format(opt.saved_path))

        # 达到目标分数提前结束
        if opt.target_score > 0 and best_score >= opt.target_score:
            print("Target score {} reached! Stopping training early.".format(opt.target_score))
            torch.save(model, "{}/tetris".format(opt.saved_path))
            print("Training finished. Best score: {}".format(best_score))
            return

    # 训练结束保存最终模型
    torch.save(model, "{}/tetris".format(opt.saved_path))
    print("Training finished. Best score: {}".format(best_score))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
