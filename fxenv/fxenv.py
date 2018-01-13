import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#どれくらい過去のデータを入力するか
WINDOW_SIZE = 60
#1エピソードのステップ数 1日分
STEPS = 1440 * 1
#売り買いのボジションの単位
POSITION_UNIT = 10000
#スプレッド(円)
SPREAD = 0.004

class FxEnv(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        super(FxEnv, self).__init__()

        # 0:売りポジ 1:ノーポジ 2:買いポジ
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, WINDOW_SIZE + 1)

        self.load_history()

        # エピソードの記録
        self.log_columns = ['price', 'position', 'reward']

    def _reset(self):
        self.prev_position = 0
        self.start_index = np.random.randint(low=WINDOW_SIZE, high=self.history.index.size-STEPS)
        self.current_step = 0
        self.current_index = self.start_index
        self.prev_obs = self.make_obs()
        self.log_data = pd.DataFrame(columns = self.log_columns)
        self.reward_sum = 0
        return self.prev_obs

    def _step(self, action):
        # 保有ポジション*価格変動分が報酬
        price_diff = self.history['close'][self.current_index-1] - self.history['close'][self.current_index-2]
        reward = self.prev_position * price_diff

        # 新規ポジション作成時にスプレッド分の手数料を支払う
        position = (action - 1) * POSITION_UNIT
        if position != self.prev_position:
            reward -= abs(position) * SPREAD
        self.reward_sum += reward

        self.log_data = self.log_data.append(pd.DataFrame([[
            self.history['close'][self.current_index-1],
            position,
            self.reward_sum
        ]], columns=self.log_columns, index=[self.history['date'][self.current_index-1]]))

        # 時刻を 1 unit 進め、情報を更新する
        self.current_index += 1
        self.current_step += 1
        self.prev_position = position
        self.prev_obs = self.make_obs()

        finish = self.current_step >= STEPS
        if finish:
            self.draw_chart()
        return self.prev_obs, reward, finish, {}

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def load_history(self):
        file = './history/DAT_ASCII_USDJPY_M1_2017.csv'
        history_all = pd.read_csv(file,
                                  sep=';',
                                  names=['date','open','high','low','close','v'],
                                  parse_dates=['date'])
        self.history = history_all[['date','close']]

    def make_obs(self):
        prices = self.history['close'][self.current_index-WINDOW_SIZE:self.current_index].values
        normal_values = prices - prices.min()
        normal_values = normal_values / normal_values.max()
        pos = ((self.prev_position / POSITION_UNIT) + 1 ) /2
        obs =  np.concatenate([[pos], normal_values])
        return obs

    def draw_chart(self):
        ax1, ax2, ax3  = self.log_data.plot(subplots=True)
        plt.show()
