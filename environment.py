from random import randrange as rand
import random
import numpy as np
import utils
from Args import Args
from collections import deque


class Tetris(object):
    def __init__(self):
        self.random_rotation = True
        self.random_position_x = True
        self.stone_set = Args.stone_shapes3
        self.init_game()

    def init_game(self, seed_start = None):
        self.seed = seed_start
        self.new_board()
        self.new_stone()
        self.score = Args.default_reward
        self.gameover = False
        self.previous_actions = deque([3, 3, 3, 3, 3, 3], maxlen=6)


    def new_board(self):
        self.board = np.zeros((Args.rows, Args.cols)) #To bool?

    def new_stone(self):
        if self.seed:
            self.seed+=1
            random.seed(self.seed)
        self.stone = random.choice(self.stone_set)
        self.stone_x = 1
        self.stone_y = 0

        if self.random_rotation:
            rotate = random.random()
            if rotate < 0.75:
                self.rotate()
                if rotate < 0.5:
                    self.rotate()
                    if rotate < 0.25:
                        self.rotate()

        if self.random_position_x:
            self.stone_x = random.randint(0, Args.cols - len(self.stone[0]))
            #self.stone_x = random.randint(1, Args.cols-len(self.stone[0])-1)
        else:
            self.stone_x = int(Args.cols / 2 - len(self.stone[0]) / 2)

        if np.any(self.board[2,:]): # check_collision()
            self.gameover = True


    def state(self):
        # appends the stone and returns board
        board_state = np.copy(self.board)
        for row in range(self.stone.shape[0]):
            for column in range(self.stone.shape[1]):
                if self.stone[row, column]:
                    board_state[row+self.stone_y, column+self.stone_x] = 1
        return board_state

    def move(self, direction):
        prev_x = np.copy(self.stone_x)
        self.stone_x = self.stone_x + direction
        if self.check_collision():
            self.score = Args.reward_for_unnecessary_move
            self.stone_x = prev_x
            if Args.end_game_on_unnecessary:
                self.gameover = True

    def rotate(self):
        prev_stone = np.copy(self.stone)
        self.stone = np.rot90(self.stone)
        if self.check_collision():
            self.score = Args.reward_for_unnecessary_move
            self.stone = prev_stone
            if Args.end_game_on_unnecessary:
                self.gameover = True

    def drop(self):
        while True:
            prev_y = np.copy(self.stone_y)
            self.stone_y += 1
            if self.check_collision():
                self.stone_y = prev_y
                self.board = self.state()
                self.score_and_clear_rows()
                break
        self.new_stone()

    def drop_one(self):
        prev_y = np.copy(self.stone_y)
        self.stone_y += 1
        if self.check_collision():
            self.stone_y = prev_y
            self.board = self.state()
            self.score_and_clear_rows()
            self.new_stone()

    def check_collision(self):
        if self.stone_y + self.stone.shape[0] > Args.rows:
            return True
        if self.stone_x < 0 or self.stone_x + self.stone.shape[1] > Args.cols:
            return True
        for row in range(self.stone.shape[0]):
            for column in range(self.stone.shape[1]):
                if self.stone[row, column] and self.board[row+self.stone_y, column+self.stone_x]:
                    return True
        return False

    def score_and_clear_rows(self):
        rows_to_delete = []
        for row_number in range(self.stone.shape[0]):
            # loops the rows where the stone is placed
            row_number += self.stone_y
            if np.all(self.board[row_number]):
                rows_to_delete.append(row_number)
        cleared_rows = len(rows_to_delete)
        self.score = Args.linescores[cleared_rows]
        if cleared_rows > 0:
            #print('CLEARED ROWS!!!!!!!', cleared_rows)
            self.board = np.delete(self.board, rows_to_delete, axis=0)
            self.board = np.concatenate((np.zeros((cleared_rows, Args.cols)), self.board), axis=0)

    def step(self, action):
        self.score = Args.default_reward
        #if action == 0: print("rotate")
        #elif action == 1: print("left")
        #elif action == 2: print("rigth")
        #else: print("drop")

        if not Args.enable_rotate:
            action += 1

        self.previous_actions.append(action)
        if self.previous_actions.count(1)==3 and self.previous_actions.count(2)==3:
            #oscillating
            self.gameover = True

        if 0 == self.previous_actions[0] == self.previous_actions[1] == self.previous_actions[2]\
                ==self.previous_actions[3] ==self.previous_actions[3]:
            self.gameover = True

        if action == 0:
            self.score = Args.rotation_reward
            self.rotate()
        elif action == 1:
            self.move(-1)
        elif action == 2:
            self.move(1)
        else:
            self.drop()
        #if random.random() < 0.02:
            #occasionally drop the piece by one
        #    self.drop_one()
        if self.gameover:
            self.score += Args.reward_for_losing

        return self.score


class tetris_simulation(object):
    def __init__(self):
        self.tetris = Tetris()
        self.action_space = Args.actions
        self.observation_space = (Args.rows, Args.cols, 1)
        self.step_count = 0
        self.cleared_row_count = 0

        self.random_rotation_prob = .5
        self.random_position_prob = .5
        self.stone_sets = [Args.stone_shapes3]

    def reset(self, seed_start = False):
        self.tetris.random_position_x = False
        self.tetris.random_rotation = False
        if random.random() < self.random_position_prob:
            self.tetris.random_position_x = True
        if random.random() < self.random_rotation_prob:
            self.tetris.random_rotation = True
        self.tetris.stone_set = random.choice(self.stone_sets)
        self.tetris.init_game(seed_start)
        state = self.tetris.state()[:, :, np.newaxis]
        self.step_count = 0
        return state

    def step(self, action):
        reward = self.tetris.step(action)
        state = self.tetris.state()[:, :, np.newaxis]
        gameover = self.tetris.gameover
        self.step_count += 1
        if self.step_count > Args.max_steps:
            reward = 0
            gameover = True
        if reward >= 1:
            self.cleared_row_count += reward

        #if gameover:  # loops around without quitting the game
        #    #gameover = False
        #    return self.reset(), reward, gameover, {}


        return state, reward, gameover, {}

    def render(self, mode='human'):
        utils.show_board(self.tetris.state())



if __name__ == "__main__":
    tr = tetris_simulation()
    s = tr.reset()
    utils.show_board(s)

    test_actions = [
        2,2,2,2,2,2,2,2
    ]

    for action in test_actions:
        s, reward, done, _ = tr.step(action)
        tr.render()
        print(reward)
        if done:
            break
