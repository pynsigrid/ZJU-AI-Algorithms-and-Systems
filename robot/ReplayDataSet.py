import random
from collections import namedtuple
import numpy as np
import copy
from torch.utils.data import DataLoader
from Maze import Maze


class ReplayDataSet(object):
    def __init__(self, max_size):
        super(ReplayDataSet, self).__init__()
        #nametuple创建一个命名的元组，名称为"Row"，元组内元素按顺序分别名为"state", "action_index", "reward", "next_state", "is_terminal"
        self.Row = namedtuple("Row", field_names=["state", "action_index", "reward", "next_state", "is_terminal"])
        self.max_size = max_size
        self.Experience = {}        #创建Experience字典
        self.full_dataset = []      #创建full_dataset列表

    def add(self, state, action_index, reward, next_state, is_terminal):            #向row中加入一个新的位置的新的动作，并更新Experience
        if len(self.Experience) == self.max_size:
            self.Experience.popitem()  # 超越内存最大长度时需要清理空间，具体的删除方式还需改动

        key = (state, action_index)
        if self.Experience.__contains__(key):
            return
        else:
            new_row = self.Row(list(state), action_index, reward, list(next_state), is_terminal)    #将输入的元组state与next_state转化为列表，如(0,1)变成[0,1]
            self.Experience.update({key: new_row})

    def random_sample(self, batch_size):                                    #根据batch_size选取Experience中的经验
        if len(self.Experience) < batch_size:
            print("the amount of experiences is to few")
            return
        else:
            samples = random.sample(list(self.Experience.values()), batch_size)
            state = []
            action_index = []
            reward = []
            next_state = []
            is_terminal = []
            for single_sample in samples:
                state.append(single_sample.state)
                action_index.append([single_sample.action_index])
                reward.append([single_sample.reward])
                next_state.append(single_sample.next_state)
                is_terminal.append([single_sample.is_terminal])
            return np.array(state), np.array(action_index, dtype=np.int8), np.array(reward), np.array(
                next_state), np.array(
                is_terminal, dtype=np.int8)

    def build_full_view(self, maze: Maze):
        """
            金手指，获取迷宫全图视野的数据集
            :param maze: 由Maze类实例化的对象
        """
        maze_copy = copy.deepcopy(maze)
        maze_size = maze_copy.maze_size
        actions = ["u", "r", "d", "l"]
        for i in range(maze_size):
            for j in range(maze_size):
                state = (i, j)
                newState = [0 for k in range(maze_size*maze_size)]  #
                newState[i*maze_size+j] = 1                         #
                if state == maze_copy.destination:
                    continue
                for action_index, action in enumerate(actions):
                    maze_copy.robot["loc"] = state
                    reward = maze_copy.move_robot(action)
                    next_state = maze_copy.sense_robot()
                    new_next_state = [0 for k in range(maze_size*maze_size)]                                #
                    new_next_state[next_state[0]*maze_size+next_state[1]] = 1                               #
                    is_terminal = 1 if next_state == maze_copy.destination or next_state == state else 0    
                    self.add(tuple(newState), action_index, reward, tuple(new_next_state), is_terminal)     #
        self.full_dataset = list(self.Experience.values())

    # def build_full_view(self, maze: Maze):
    #     """
    #         金手指，获取迷宫全图视野的数据集
    #         :param maze: 由Maze类实例化的对象
    #     """
    #     maze_copy = copy.deepcopy(maze)
    #     maze_size = maze_copy.maze_size
    #     actions = ["u", "r", "d", "l"]
    #     for i in range(maze_size):
    #         for j in range(maze_size):
    #             state = (i, j)
                
    #             if state == maze_copy.destination:
    #                 continue
    #             for action_index, action in enumerate(actions):
    #                 maze_copy.robot["loc"] = state
    #                 reward = maze_copy.move_robot(action)
    #                 next_state = maze_copy.sense_robot()
    #                 is_terminal = 1 if next_state == maze_copy.destination or next_state == state else 0
    #                 self.add(state, action_index, reward, next_state, is_terminal)
    #     self.full_dataset = list(self.Experience.values())

    def __getitem__(self, item):
        state = self.full_dataset[item].state
        action_index = self.full_dataset[item].action_index
        reward = self.full_dataset[item].reward
        next_state = self.full_dataset[item].next_state
        is_terminal = self.full_dataset[item].is_terminal
        return np.array(state), np.array([action_index], dtype=np.int8), np.array([reward]), np.array(
            next_state), np.array([is_terminal], dtype=np.int8)

    def __len__(self):
        return len(self.Experience)


if __name__ == "__main__":
    memory = ReplayDataSet(1e3)
    maze1 = Maze(5)
    memory.build_full_view(maze1)
    print(len(memory))
    # memory_loader = DataLoader(memory, batch_size=5)
    # for idx, content in enumerate(memory_loader):
    #     print(idx)
    #     print(content)
    #     break
