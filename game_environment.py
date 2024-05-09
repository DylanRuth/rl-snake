import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

class SnakeNumpy:
    def __init__(self, board_size=10, frames=2, games=10, start_length=2, seed=42,
                 max_time_limit=298, frame_mode=False, obstacles=False, version=''):
        self._value = {'snake':1, 'board':0, 'food':3, 'head':2, 'border':4}
        self._actions = {-1:-1, 0:0, 1:1, 2:2, 3:3, 4:-1}
        self._n_actions = 4
        self._board_size = board_size
        self._n_frames = frames
        self._n_games = games
        self._rewards = {'out':-1, 'food':1, 'time':0, 'no_food':0}
        self._start_length = 2
        self._max_time_limit = max_time_limit
        self._board = deque(maxlen = self._n_frames)
        self._action_conv = np.zeros((3,3,self._n_actions), dtype=np.uint8)
        self._action_conv[1,0,0] = 1
        self._action_conv[2,1,1] = 1
        self._action_conv[1,2,2] = 1
        self._action_conv[0,1,3] = 1
        self._termination_reason_dict = {
            'game_end'        : 1,
            'collision_wall'  : 2,
            'collision_self'  : 3,
            'time_up'         : 4,
            'time_up_no_food' : 5
        }
        self._frame_mode = frame_mode
        self._obstacles = obstacles
        self._version = version

    def _queue_to_board(self):
        board = np.stack([x for x in self._board], axis=3)
        return board.copy()

    def _random_seq(self):
        seq = np.arange(1,1+self._board_size**2, dtype=np.uint16)
        self._seq = np.zeros((self._n_games,self._board_size,self._board_size))
        for i in range(self._n_games):
            np.random.shuffle(seq)
            self._seq[i] = seq.copy().reshape((1,self._board_size,self._board_size))

    def _random_snake(self):
        strides = self._board_size - 2 - self._start_length + 1
        total_boards = strides * (self._board_size-2) * 4
        self._body_random = np.zeros((total_boards,
                                      self._board_size, self._board_size), 
                                      dtype=np.uint16)

        self._head_random = self._body_random.copy()
        self._direction_random = np.zeros((total_boards,), dtype=np.uint8)

        for i in range(strides):
            idx1 = np.arange(0+i*(self._board_size-2),0+(i+1)*(self._board_size-2), dtype=np.uint8)
            idx2 = np.arange(1,self._board_size-1, dtype=np.uint8)
            self._body_random[idx1,idx2,i+1:i+1+self._start_length-1] = (np.arange(self._start_length-1, dtype=np.uint16)+1)
            self._head_random[idx1,idx2,i+1+self._start_length-1] = 1

        idx1 = np.arange(total_boards//4, (total_boards//4)*2)
        idx2 = np.arange(total_boards//4)
        self._body_random[idx1,:,::-1] = self._body_random[idx2,:,:].copy()
        self._head_random[idx1,:,::-1] = self._head_random[idx2,:,:].copy()
        self._direction_random[idx1] = 2
        
        idx1 = np.arange(total_boards//4, (total_boards//4)*2)
        for i in idx1:
            self._body_random[i+(total_boards//4),:,:] = self._body_random[i,::-1,:].copy().T
            self._head_random[i+(total_boards//4),:,:] = self._head_random[i,::-1,:].copy().T
        self._direction_random[idx1 + (total_boards//4)] = 3

        idx1 = np.arange((total_boards//4)*3, (total_boards//4)*4)
        idx2 = np.arange((total_boards//4)*2, (total_boards//4)*3)
        self._body_random[idx1,::-1,:] = self._body_random[idx2,:,:].copy()
        self._head_random[idx1,::-1,:] = self._head_random[idx2,:,:].copy()
        self._direction_random[idx1] = 1

    def _random_board(self):
        if(not self._obstacles):
            self._border_random = self._value['board'] * np.ones((self._board_size-2,self._board_size-2), 
                                                          dtype=np.uint8)
            self._border_random = np.pad(self._border_random, 1, mode='constant',
                                  constant_values=self._value['border'])\
                              .reshape(1,self._board_size,self._board_size)
            self._border_random = np.zeros((self._n_games, self._board_size, self._board_size)) \
                            + self._border_random
        else:
            with open('models/{:s}/obstacles_board'.format(self._version), 'rb') as f:
                self._border_random = pickle.load(f)
            self._border_random *= self._value['border']

    def _calculate_board_wo_food(self):
        board = self._border + (self._body > 0)*self._value['snake'] + \
                self._head*self._value['head']
        return board.copy()

    def _calculate_board(self):
        board = self._calculate_board_wo_food() + self._food*self._value['food']
        return board.copy()

    def _weighted_sum(self, w, x1, x2):
        w = w.reshape(-1,1,1)
        return (w*x1 + (1-w)*x2).copy()

    def _set_first_frame(self):
        board = self._calculate_board()
        self._board[0] = self._weighted_sum((1-self._done), board, self._board[0])

    def _reset_frames(self, f):
        board = self._calculate_board_wo_food()
        for i in range(1, len(self._board)):
            self._board[i][f] = board[f]

    def print_game(self):
        board = self._queue_to_board()
        fig, axs = plt.subplots(self._n_games, self._n_frames)
        if(self._n_games == 1 and self._n_frames == 1):
            axs.imshow(board[0], cmap='gray')
        elif(self._n_games == 1):
            for i in range(self._n_frames):
                axs[i].imshow(board[0,:,:,i], cmap='gray')
        elif(self._n_frames == 1):
            for i in range(self._n_games):
                axs[i].imshow(board[i,:,:,0], cmap='gray')
        else:
            for i in range(self._n_games):
                for j in range(self._n_frames):
                    axs[i][j].imshow(board[i,:,:,j], cmap = 'gray')
        plt.show()

    def get_board_size(self):
        return self._board_size

    def get_n_frames(self):
        return self._n_frames

    def get_head_value(self):
        return self._value['head']

    def get_values(self):
        return self._value

    def get_legal_moves(self):
        a = np.ones((self._n_games, self._n_actions), dtype=np.uint8)
        a[np.arange(self._n_games), (self._snake_direction-2)%4] = 0
        return a.copy()

    def reset(self, stateful=False):
        if(stateful and len(self._board)>0):
            return self._queue_to_board()

        self._random_seq()
        self._random_snake()

        self._random_board()
        random_indices = np.random.choice(self._border_random.shape[0], self._n_games)
        self._border = self._border_random[random_indices].copy()

        self._food = np.zeros((self._n_games, self._board_size, self._board_size), dtype=np.uint8)
        if(not self._obstacles):
            random_indices = np.random.choice(self._body_random.shape[0], self._n_games)
        else:
            random_indices = np.zeros((self._n_games,), dtype=np.int16)
            for i in range(self._n_games):
                random_indices_mask = ((self._body_random + self._head_random) * self._border[i])\
                                        .sum(axis=(1,2)) == 0
                random_indices_mask = random_indices_mask/random_indices_mask.sum()
                random_indices[i] = int(np.random.choice(np.arange(self._body_random.shape[0]), 
                                                  1, p=random_indices_mask))
        self._body, self._head, self._snake_direction = \
                                self._body_random[random_indices].copy(),\
                                self._head_random[random_indices].copy(),\
                                self._direction_random[random_indices].copy()

        self._snake_length = self._start_length * np.ones((self._n_games), dtype=np.uint16)
        self._count_food = np.zeros((self._n_games), dtype=np.uint16)
        board = self._calculate_board()
        for _ in range(self._n_frames):
            self._board.append(board.copy())
        
        self._get_food()
        
        self._time = np.zeros((self._n_games), dtype=np.uint16)
        self._done = np.zeros((self._n_games,), dtype=np.uint8)
        self._cumul_rewards = np.zeros((self._n_games,), dtype=np.int16)
        self._set_first_frame()
        return self._queue_to_board()

    def _soft_reset(self):
        f = (self._done == 1)
        fsum = self._done.sum()
        self._food[f] = np.zeros((fsum, self._board_size,self._board_size),
                                 dtype=np.uint8)

        random_indices = np.random.choice(np.arange(self._border_random.shape[0]), fsum)
        self._border[f] = self._border_random[random_indices].copy()

        if(not self._obstacles):
            random_indices = np.random.choice(np.arange(self._body_random.shape[0]), fsum)
        else:
            random_indices = np.zeros((fsum,), dtype=np.int16)
            i = 0
            for i1 in range(self._done.shape[0]):
                if(self._done[i1] == 1):
                    random_indices_mask = ((self._body_random + self._head_random) * self._border[i1])\
                                            .sum(axis=(1,2)) == 0
                    # convert to probabilities for the random choice function
                    random_indices_mask = random_indices_mask/random_indices_mask.sum()
                    random_indices[i] = int(np.random.choice(np.arange(self._body_random.shape[0]), 
                                                  1, p=random_indices_mask))
                    i += 1

        self._body[f], self._head[f], self._snake_direction[f] = \
                        self._body_random[random_indices].copy(),\
                        self._head_random[random_indices].copy(),\
                        self._direction_random[random_indices].copy()

        self._snake_length[f] = self._start_length
        self._time[f] = 0
        self._done[f] = 0
        self._cumul_rewards[f] = 0
        self._get_food()
        self._set_first_frame()
        self._reset_frames(f)

        if(np.random.random() < 0.01):
            self._random_seq()

    def get_num_actions(self):
        return self._n_actions

    def _action_map(self, action):
        return self._actions[action]

    def _get_food(self):
        food_pos = ((self._border + self._body + self._head) == self._value['board']) * self._seq
        m = food_pos.max((1,2)).reshape(self._n_games,1,1)
        food_pos = ((food_pos == m) & (food_pos > self._value['board']))
        self._food = self._weighted_sum(1-self._food.max((1,2)), food_pos, self._food).astype(np.uint8)

    def _get_new_direction(self, action, current_direction):
        new_dir = current_direction.copy()
        f = (np.abs(action - current_direction) != 2) & (action != -1)
        new_dir[f] = action[f]
        return new_dir.copy()

    def _get_new_head(self, action, current_direction):
        action = self._get_new_direction(action, current_direction)
        one_hot_action = np.zeros((self._n_games,1,1,self._n_actions), dtype=np.uint8)
        one_hot_action[np.arange(self._n_games), :, :, action] = 1
        hstr = self._head.strides
        new_head = np.lib.stride_tricks.as_strided(self._head, 
                       shape=(self._n_games,self._board_size-3+1,self._board_size-3+1,3,3),
                       strides=(hstr[0],hstr[1],hstr[2],hstr[1],hstr[2]),
                       writeable=False)
        new_head = (np.tensordot(new_head,self._action_conv) * one_hot_action).sum(3)
        new_head1 = np.zeros(self._head.shape, dtype=np.uint8)
        new_head1[:,1:self._board_size-1,1:self._board_size-1] = new_head
        return new_head1.copy()

    def step(self, action):
        reward, can_eat_food, termination_reason, new_head \
                    = self._check_if_done(action)
        self._move_snake(action, can_eat_food, new_head)
        self._snake_direction = self._get_new_direction(action, self._snake_direction)
        self._time += (1-self._done)

        self._cumul_rewards += reward
        info = {'time':self._time.copy(), 'food':self._count_food.copy(),
                'termination_reason':termination_reason.copy(),
                'length':self._snake_length.copy(),
                'cumul_rewards':self._cumul_rewards.copy()}
        done_copy = self._done.copy()
        if(self._frame_mode):
            self._soft_reset()
        next_legal_moves = self.get_legal_moves()
        return self._queue_to_board(), reward.copy(), done_copy.copy(),\
                info, next_legal_moves.copy()

    def _get_food_reward(self, f):
        return self._rewards['food']

    def _get_death_reward(self, f):
        return self._rewards['out']

    def _check_if_done(self, action):
        reward, can_eat_food, termination_reason = \
                            self._rewards['time'] * np.ones((self._n_games,), dtype=np.int16),\
                            np.zeros((self._n_games,), dtype=np.uint8),\
                            np.zeros((self._n_games), dtype=np.uint8)
        done_copy = self._done.copy()
        new_head = self._get_new_head(action, self._snake_direction)
        f1 = (self._snake_length == (self._board_size-2)**2)
        self._done[f1] = 1
        reward[f1] += self._get_food_reward(f1)
        termination_reason[f1] = 1
        f2 = ((new_head.sum((1,2)) == 0) | \
                ((new_head * self._border).sum((1,2)) > 0))
        f = f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward(f)
        termination_reason[f] = 2
        body_head_sum = (self._body * new_head).sum((1,2))
        f3 = (body_head_sum > 0) & ~(body_head_sum == 1)
        f = f3 & ~f2 & ~f1
        self._done[f] = 1
        reward[f] = self._get_death_reward(f)
        termination_reason[f] = 3

        f4 = ((self._food * new_head).sum((1,2)) == 1)
        f = f4 & ~f3 & ~f2 & ~f1
        reward[f] += self._get_food_reward(f)
        can_eat_food[f] = 1
        if(self._max_time_limit != -1):
            f5 = (self._time >= self._max_time_limit)
            f = f5 & ~f4 & ~f3 & ~f2 & ~f1
            self._done[f] = 1
            termination_reason[f] = 4
            if(self._rewards['no_food'] != 0):
                f6 = (self._snake_length == self._start_length)
                f = f6 & ~f5 & ~f4 & ~f3 & ~f2 & ~f1
                termination_reason[f] = 5
                reward[f] += self._rewards['no_food']
        reward[done_copy == 1] = 0

        return reward.copy(), can_eat_food.copy(), termination_reason.copy(), new_head.copy()

    def _move_snake(self, action, can_eat_food, new_head):
        new_body = self._body.copy()
        body_max = self._body.max((1,2))
        self._body = (self._done).reshape(-1,1,1)*self._body + \
                     ((1-self._done)*can_eat_food).reshape(-1,1,1)*(self._body+(body_max+1).reshape(-1,1,1)*self._head) +\
                     ((1-self._done)*(1-can_eat_food)).reshape(-1,1,1)*(new_body+body_max.reshape(-1,1,1)*self._head)
        self._head = self._weighted_sum(self._done, self._head, new_head)
        if(can_eat_food.sum()>0):
            self._snake_length[can_eat_food == 1] += 1
            self._count_food[can_eat_food == 1] += 1
            self._food = self._weighted_sum((1-can_eat_food), self._food, 0)
            self._get_food()
        self._board.appendleft(self._board[0].copy())
        self._set_first_frame()
