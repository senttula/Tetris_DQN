from random import randrange as rand
import random
import numpy as np
import utils
from Args import Args
from collections import deque


class RingBuffer(object):
    # keras rl/memory
    def __init__(self, maxlen, init_value=None):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [init_value for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return element of buffer at specific index

        # Argument
            idx (int): Index wanted

        # Returns
            The element of buffer at given index
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def count(self, i):
        return self.data.count(i)

    def append(self, v):
        """Append an element to the buffer

        # Argument
            v (object): Element to append
        """
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

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
        #self.previous_actions = deque([3, 3, 3, 3, 3, 3, 3, 3], maxlen=8)
        self.previous_actions = RingBuffer(8, init_value=3)


    def new_board(self):
        self.board = np.ones((Args.rows, Args.cols), dtype=np.int16) #To bool?

    def new_stone(self):
        if self.seed:
            self.seed+=1
            random.seed(self.seed)
        self.stone = random.choice(self.stone_set)

        self.stone = np.logical_not(self.stone).astype(np.int16)

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

        if not np.all(self.board[2,:]): # check_collision()
            self.gameover = True

    def state(self):
        # appends the stone and returns board
        board_state = np.copy(self.board)
        for row in range(self.stone.shape[0]):
            for column in range(self.stone.shape[1]):
                if not self.stone[row, column]:
                    board_state[row+self.stone_y, column+self.stone_x] = 0
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
                if not self.stone[row, column] and not self.board[row+self.stone_y, column+self.stone_x]:
                    return True
        return False

    def score_and_clear_rows(self):
        rows_to_delete = []
        for row_number in range(self.stone.shape[0]):
            # loops the rows where the stone is placed
            row_number += self.stone_y
            if not np.any(self.board[row_number]):
                rows_to_delete.append(row_number)
        cleared_rows = len(rows_to_delete)
        self.score = Args.linescores[cleared_rows]
        if cleared_rows > 0:
            #print('CLEARED ROWS!!!!!!!', cleared_rows)
            self.board = np.delete(self.board, rows_to_delete, axis=0)
            self.board = np.concatenate((np.ones((cleared_rows, Args.cols)), self.board), axis=0)


    def step(self, action):
        self.score = Args.default_reward
        if not Args.enable_rotate:
            action += 1
        #if action == 0: print("rotate")
        #elif action == 1: print("left")
        #elif action == 2: print("rigth")
        #else: print("drop")



        self.previous_actions.append(action)
        if self.previous_actions.count(1)==4 and self.previous_actions.count(2)==4:
            #oscillating
            self.gameover = True

        if self.previous_actions.count(0)==8:
            # only rotating
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

        self.random_rotation_prob = 1
        self.random_position_prob = 1
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
        state = self.get_state()
        self.step_count = 0
        return state

    def get_state(self):
        st = self.tetris.state()[:, :, np.newaxis]
        return st  # +np.random.normal(0, 0.05, st.shape)

    def step(self, action):
        reward = self.tetris.step(action)
        state = self.get_state()
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

    #utils.show_board(s)
    tr.random_rotation_prob=0
    tr.random_position_prob = 0
    s = tr.reset()

    test_actions = [
        1,1,1,1,3,1,1,3,2,2,2,2,3,2,2,3,3
    ]

    for action in test_actions:
        s, reward, done, _ = tr.step(action)
        tr.render()
        if done:
            break
