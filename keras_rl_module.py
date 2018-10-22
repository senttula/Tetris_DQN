import numpy as np
import environment
from Args import Args
import utils

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Reshape, \
    GaussianNoise, Dropout, MaxPooling2D,Lambda, LocallyConnected2D
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.convolutional import Conv2D, UpSampling2D, AveragePooling2D
from keras import backend as K
from collections import deque

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
import time
import random
from keras import backend as K


#def l0_reg(weight_matrix=None):
#    #if weight_matrix:
#        return K.sum(0.1 * K.sqrt(weight_matrix))
#    #else:
#    #    print("#")


class lhalf_reg(regularizers.Regularizer):
    def __init__(self, l0=0.):
        self.l_value = K.cast_to_floatx(l0)
        self.remove_infinte_gradient = K.cast_to_floatx(0.00001)

    def __call__(self, x):
        #regularization = 0.
        regularization = K.sum(self.l_value * K.sqrt(K.abs(x)+self.remove_infinte_gradient))
        return regularization

    def get_config(self):
        return {'l0': float(self.l_value)}

def make_model():
    model = Sequential()

    model.add(Reshape((Args.rows, Args.cols, 1), input_shape=(1, Args.rows, Args.cols, 1)))

    model.add(Lambda(lambda x: 2 * x - 1.))

    #model.add(Conv2D(8, (3, 3), use_bias=False, strides=(1, 1), padding='same',
    #                 kernel_regularizer=lhalf_reg(10 ** -5),
    #                 activation='relu'
    #                 ))
#
    #model.add(Conv2D(8, (3, 3), use_bias=False, strides=(1, 1), padding='same',
    #                 kernel_regularizer=lhalf_reg(10 ** -5),
    #                 activation='relu'
    #                 ))
#
    #model.add(Conv2D(8, (3, 3), use_bias=False, strides=(1, 1), padding='same',
    #                 kernel_regularizer=lhalf_reg(10 ** -5),
    #                 activation='relu'
    #                 ))
#
    #model.add(Conv2D(8, (3, 3), use_bias=True, strides=(1, 1), padding='same',
    #                 kernel_regularizer=lhalf_reg(10 ** -5),
    #                 bias_regularizer=lhalf_reg(10 ** -5),
    #                 activation='relu'
    #                 ))



    model.add(Conv2D(16, (7, 7), use_bias=True, strides=(2, 2), padding='same',
                     kernel_regularizer=regularizers.l1(10 ** -5),
                     bias_regularizer=regularizers.l1(10 ** -5),
                     activation='relu'
                     ))


    model.add(Flatten())

    model.add(Dense(32, activation='relu', use_bias=True,
                    kernel_regularizer=regularizers.l1(10 ** -5),
                    bias_regularizer=regularizers.l1(10 ** -5),
                    ))

    model.add(Dense(Args.actions, name='to_actions', use_bias=True,
                    kernel_regularizer=regularizers.l1(10 ** -5),
                    bias_regularizer=regularizers.l1(10 ** -4),
                    ))

    model.summary()

    return model


class main():
    def __init__(self):

        self.env = environment.tetris_simulation()

        self.model = make_model()
        memory = SequentialMemory(limit=100000, window_length=1)
        # policy = EpsGreedyQPolicy(.15)
        policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=self.model, nb_actions=Args.actions, memory=memory, nb_steps_warmup=5000,
                            target_model_update=10000, policy=policy, batch_size=64
                            , enable_double_dqn=False, enable_dueling_network=False,
                            custom_model_objects={'lhalf_reg': lhalf_reg}
                            )
        self.dqn.compile(Adam(lr=0.001))


    def do(self):

        if Args.load_file:
            try:
                self.dqn.load_weights(Args.load_file)
            except:
                print("NOT LOADED")


        #self.dqn.model.load_weights('dqn_weights2.h5f', by_name=True)
        #self.dqn.update_target_model_hard()

        print('pre evaluation:')
        if self.evaluate(stone_sets=[Args.stone_shapes1, Args.stone_shapes2, Args.stone_shapes3], min_reward=99, episodes=50):
            print('pre evaluation passed, training might not be needed')
        else:
            print('starting training')

        self.env.stone_sets = [Args.stone_shapes1]

        show_sample_game(self.model, self.env, Args.stone_shapes1)

        #mins = 2000
        #self.dqn.fit(self.env, nb_steps=10000*mins, visualize=False, verbose=1, log_interval=50000)

        #self.dqn.save_weights('pitk.h5f', overwrite=True)

        stones = [Args.stone_shapes1, Args.stone_shapes2, Args.stone_shapes3]
        for i in range(1000):
                print('loop: ', i)
                if self.train_until_passes_evaluation(min_reward=99, stone_sets=[stones[2]]):
                    if self.train_until_passes_evaluation(min_reward=99, stone_sets=[stones[1]]):
                        if self.train_until_passes_evaluation(min_reward=99, stone_sets=stones[1:3]):
                            if self.train_until_passes_evaluation(min_reward=99,stone_sets=stones):
                                break


        print('finished')
    
        self.dqn.test(self.env, nb_episodes=3, visualize=True)

    def evaluate(self, stone_sets, min_reward=0, episodes=10):
        #print('last actions in a sample game: ', self.evaluate_actions(stone_sets[0]))
        boolean_list = []
        self.env.random_rotation_prob = 1
        self.env.random_position_prob = 0

        for stone_set in stone_sets:
            if min_reward == 99:
                wanted_reward = calculate_possible_reward(stone_set)
            else:
                wanted_reward = min_reward

            self.env.stone_sets = [stone_set]

            history = self.dqn.test(self.env, nb_episodes=episodes, visualize=False, verbose=0).history

            step_count_list = history['nb_steps']
            step_count_avg = sum(step_count_list) / episodes

            reward_list = np.array(history['episode_reward'])
            reward_avg = np.sum(reward_list) / episodes
            reward_std = np.std(reward_list)

            print(f'evaluation: if {round(reward_avg, 3)} > {wanted_reward}, reward deviation: {round(reward_std, 2)}, step count: '
                  f'{step_count_avg}, stone_set length: {len(stone_set)}, ', end = '')
            if reward_avg > wanted_reward:
                print('PASS')
                boolean_list.append(True)
            else:
                print('FAIL')
                boolean_list.append(False)
        if all(boolean_list) == True:
            return True
        else:
            return False

    def train_set(self, stone_sets, rotation_prob = .5, position_prob = .5, nb_steps = 30000):
        t = time.time()

        self.env.random_rotation_prob = rotation_prob
        self.env.random_position_prob = position_prob
        self.env.stone_sets = stone_sets
    
        self.env.cleared_row_count = 0
        self.dqn.fit(self.env, nb_steps=nb_steps, visualize=False, verbose=1)

        #for layer in self.model.layers:
        #   wg = layer.get_weights()
        #print("last layer bias: ", wg[1])
        print("---------- cleared rows: %d, time %.1f ----------" % (self.env.cleared_row_count, time.time() - t))
        if Args.save_file:
            self.dqn.save_weights(Args.save_file, overwrite=True)

    def train_until_passes_evaluation(self, stone_sets, min_reward=0):
        # probablity_change = [maximum of i, rotation prob divider, position prob divider

        for i in range(0, 20):

            evaluate_actions(self.model, self.env, stone_sets[0])
            if i % 1 == 0:
                show_sample_game(self.model, self.env, stone_sets[0])
                weigths_print(self.model, bias=True)

            if self.evaluate(stone_sets = stone_sets, min_reward = min_reward, episodes = 10):
                print('PASSED!')
                return True
            else:
                self.train_set(stone_sets)

        print('somethings wrong, back to previous train set')
        return False


def calculate_possible_reward(stone_set):
    total_count_of_squares = 0
    for stone in stone_set:
        total_count_of_squares += np.sum(stone)
    avg_count_of_squares_per_tile = total_count_of_squares / len(stone_set)

    avg_movements = Args.cols / 4
    avg_rotations = 1

    avg_cost_of_moving = Args.default_reward * avg_movements + \
                         Args.rotation_reward * avg_rotations + Args.linescores[0]

    avg_move_pre_stone = avg_movements + avg_rotations + 1
    avg_stones_drop = Args.max_steps // avg_move_pre_stone  # rounded down
    avg_squares_drop = avg_stones_drop * avg_count_of_squares_per_tile

    possible_lines_cleared_avg = avg_squares_drop // Args.cols  # rounded down

    estimate = possible_lines_cleared_avg + avg_cost_of_moving

    return round(estimate, 1)

def evaluate_actions(model, env, stone_set):
    #returns actions of last 10 moves
    env.random_rotation_prob = 1
    env.random_position_prob = 0
    env.stone_sets = [stone_set]
    state = env.reset()
    actions= deque([], maxlen=10)
    for i in range(Args.max_steps):
        #utils.show_board(state)
        qval = model.predict(state[np.newaxis,np.newaxis,:,:,:])
        action = np.argmax(qval)
        state = env.step(action)[0]
        actions.append(action)
    return list(actions)

def show_sample_game(model, env, stone_set):
    # returns actions of last 10 moves
    env.random_rotation_prob = 1
    env.random_position_prob = 0
    env.stone_sets = [stone_set]
    state = env.reset()
    actions = []
    total_reward=0
    qval_list = []
    for i in range(Args.max_steps):
        qval = model.predict(state[np.newaxis, np.newaxis, :, :, :])
        action = np.argmax(qval)
        state, reward, game_over, _ = env.step(action)
        actions.append(action)
        qval_list.append(qval)
        total_reward += reward
        if game_over:
            break
    qval_list = np.array(qval_list)
    action_sublists = [actions[x:x + 20] for x in range(0, len(actions), 20)]
    info_list = ['<-- end state,  total reward '+str(round(total_reward, 3)), ]
    info_list += ['    qvals median' + str(np.round(np.median(qval_list, axis=0), 5))]
    info_list += ['    qvals mean  ' + str(np.round(np.mean(qval_list, axis=0), 5))]
    info_list += ['    qvals std   ' + str(np.round(np.std(qval_list, axis=0), 5))]
    info_list += ['    random qval ' + str(np.round((qval_list[qval_list.shape[0]//2]), 5))] # not really random, but enough for example
    info_list += action_sublists
    info_list += ['    made actions above']
    for row in range(state.shape[0]):
        for column in range(state.shape[1]):
            if state[row, column]:
                print("#", end="")
            else:
                print(".", end="")

        if len(info_list)>row:
            print('    ', info_list[row])
        else:
            print()
    return list(actions)

def weigths_print(model, bias = False, quit_on_zerosum=True):
    print('%12s%8s%8s%8s%8s%8s%10s --- weigths of layers' % ('name', 'mean', 'std', 'max', 'min', 'sum(abs)', 'mean(abs)'))
    quit_flag = False # if model is bad
    for layer in model.layers:
        wg_orig = layer.get_weights()
        if wg_orig:
            wg = wg_orig[0]
            #print(wg)
            infos = np.array([np.mean(wg), np.std(wg), np.max(wg), np.min(wg), np.sum(np.abs(wg)),
                              np.mean(np.abs(wg)), np.count_nonzero(np.round(wg, 6))/np.size(wg)])
            # print(z)
            print('%12s' % layer.name, end='  ')
            for number in infos:
                print('%7.3f' % number, end=' ')
            print()
            if quit_on_zerosum and not quit_flag and np.round(np.sum(np.abs(wg)), 3)==0:
                quit_flag = True
            if np.isnan(wg).any() and not quit_flag:
                quit_flag = True
            if bias and len(wg_orig) ==2:
                wg = wg_orig[1]
                z = np.array([np.mean(wg), np.std(wg), np.max(wg), np.min(wg), np.mean(np.abs(wg)), np.mean(np.abs(wg)),
                              np.count_nonzero((np.round(wg, 6)))/np.size(wg)])
                # print(z)
                print('%10s b' % layer.name, end='  ')
                for number in z:
                    print('%7.3f' % number, end=' ')
                print()

    if len(model.layers[-1].get_weights())==2:
        print ('last bias: ', model.layers[-1].get_weights()[1])
    if quit_flag:
        print('layer sum(abs) ~0 or encountered NaN, quitting')
        quit()


if __name__ == "__main__":
    mn = main()
    mn.do()