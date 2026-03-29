#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import torch

from core.deep_q_network import DuelingDQN
from core.game import Tetris


def get_args():
    parser = argparse.ArgumentParser("""使用训练好的模型游玩 Tetris""")
    parser.add_argument("--model", type=str, default="trained_models/tetris_best",
                        help="模型文件路径")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30, help="回放帧率")
    parser.add_argument("--games", type=int, default=1, help="连续玩几局")
    args = parser.parse_args()
    return args


def play(opt):
    if not os.path.exists(opt.model):
        print("模型文件不存在: {}".format(opt.model))
        return

    model = torch.load(opt.model, map_location=lambda s, _: s, weights_only=False)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)

    for game_num in range(opt.games):
        env.reset()
        done = False
        while not done:
            next_steps = env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)

            with torch.no_grad():
                predictions = model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]

            _, done = env.step(action, render=True)

        print("Game {}/{}, Score: {}, Tetrominoes: {}, Cleared lines: {}".format(
            game_num + 1, opt.games, env.score, env.tetrominoes, env.cleared_lines))

        # 等待按键或 2 秒后开始下一局
        if game_num < opt.games - 1:
            print("按任意键开始下一局...")
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = get_args()
    play(opt)
