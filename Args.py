
import numpy as np

class Args:

    ##################################################################################
    # Environment rewards
    default_reward = -0.01
    rotation_reward = -0.5
    reward_for_losing = -1
    linescores = [0.1, 1, 3, 9, 27]  # first score is for dropped piece that doesn't clear any rows
    reward_for_unnecessary_move = -1 # move that doesn't change state
    end_game_on_unnecessary = True

    ##################################################################################
    # Learning parameters

    replay_batch = 1024*8
    memory_size = replay_batch*4
    gamma = 0.99

    epochs = 2
    max_steps = 100

    learning_rate=0.0005

    load_file = None
    save_file = 'weights.h5f'

    ##################################################################################
    # Tetris environment
    cols = 10
    rows = 12

    random_rotation = True
    random_position_x = False

    enable_rotate = True  # is rotation enabled, actions change from 3 to 4
    actions = 3+enable_rotate


    stone_shapes1 = [
        np.array([[0, 1, 0],
                  [1, 1, 1]]),

        np.array([[0, 1, 1],
                  [1, 1, 0]]),

        np.array([[1, 1, 0],
                  [0, 1, 1]]),

        np.array([[1, 0, 0],
                  [1, 1, 1]]),

        np.array([[0, 0, 1],
                  [1, 1, 1]]),

        np.array([[1, 1, 1, 1]]),

        np.array([[1, 1],
                  [1, 1]])
    ]

    stone_shapes2 = [
        np.array([[1]]),

        np.array([[1, 1]]),

        np.array([[1, 0],
                  [1, 1]]),

        np.array([[1, 1, 1]]),
    ]

    stone_shapes3 = [
        np.array([[1, 1]])
    ]


    stone_shapes = [
        np.array([[1, 1]])
    ]















