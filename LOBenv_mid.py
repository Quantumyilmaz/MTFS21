# Author: Ahmet Ege Yilmaz
# Year: 2021
# The code in this file reproduces the reinforcement learning environment in "Deep Reinforcement Learning for Active High Frequency Trading" by Antonio Briola, Jeremy Turiel, Riccardo Marcaccioli, Tomaso Aste.

import numpy as np
import gym
from gym import spaces
import math
from sklearn.decomposition import PCA

LOB_depth = 5
resSize = 100
initLen=0

vol_log_base = 1e10
price_log_base = 1e2

class LOBEnv_mid(gym.Env):

    def __init__(self, LOB, scenario=201,prices_enabled=True,window_size=10,spread=0,reservoir_sample=None,pca_sample=None,normalize_data=False,normalizer={},id_=None,verbose=True):
        if reservoir_sample is not None:
            LOB = LOB.iloc[initLen:]

        self.__id = '_'+str(id_) if id_ is not None else ''
        if verbose:
            print(f'Initializing LOBenv_mid{self.__id}...')
        assert scenario>200
        assert bool(normalize_data) +  bool(normalizer) < 2
        super(LOBEnv_mid, self).__init__()
        
        self.scenario = scenario
        self.prices_enabled = prices_enabled
        self.window_size = window_size
        self.spread = spread
        self.use_pca = False
        self.using_reservoir = False
        self.reduced = None
        
        self.position_space = spaces.Discrete(3) # ["N","L","S"]
        self.action_space = spaces.Discrete(4) # {sell, stay, buy, daily_stop_loss}
        self.dtype = LOB.dtypes[0]

        input_length = LOB_depth*2+prices_enabled
    
        #Just take volumes
        
        lob = LOB.copy()
        self.__LOB_volumes = lob[lob.columns[lob.columns.str.contains("size")]]
        self.__LOB_prices = lob[lob.columns[~lob.columns.str.contains("size")]]
        self.__LOB_mid_prices = (lob["bid1"]+lob["ask1"])/2

        if reservoir_sample is not None:
            
            input_length += resSize

            self.reservoir_sample = reservoir_sample[21:,initLen:]
            self.using_reservoir = True

            self.LOB_volumes = self.__LOB_volumes.applymap(lambda x:math.log(x,vol_log_base))
            self.LOB_prices = self.__LOB_prices.applymap(lambda x:math.log(x,price_log_base))
            self.LOB_mid_prices = self.__LOB_mid_prices.apply(lambda x:math.log(x,price_log_base))
        else:
            self.LOB_volumes = self.__LOB_volumes.copy()
            self.LOB_prices = self.__LOB_prices.copy()
            self.LOB_mid_prices = self.__LOB_mid_prices.copy()

            # self.LOB_volumes = self.__LOB_volumes.applymap(lambda x:math.log(x,vol_log_base))
            # self.LOB_prices = self.__LOB_prices.applymap(lambda x:math.log(x,price_log_base))
            # self.LOB_mid_prices = self.__LOB_mid_prices.apply(lambda x:math.log(x,price_log_base))

        if pca_sample is not None:
            self.use_pca = True
            self.reduced = pca_sample
            input_length -= LOB_depth*2+prices_enabled
            input_length += pca_sample.shape[-1]
        
        if normalize_data:
            self.LOB_volumes /= self.LOB_volumes.max()*2
            self.LOB_prices /= self.LOB_prices.max()*2
            self.LOB_mid_prices /= self.LOB_mid_prices.max()*2    

        for key in normalizer:
            if key == "size":
                self.LOB_volumes /= normalizer[key]
            elif key == "price":
                self.LOB_prices /= normalizer[key]
                self.LOB_mid_prices /= normalizer[key]
            elif key=="subtract":
                self.LOB_volumes -= normalizer[key]
                self.LOB_prices -= normalizer[key]
                self.LOB_mid_prices -= normalizer[key]
            else:
                raise Exception("Normalizer has unknown key.")
        
        
        # self.day = LOB.index[0].date()
        self.day_ticksize = len(LOB)
         
        self.action_list = []

        self.position = 0
        self.position_list = [(0,0,0)]
        self.position_price = None
        
        self.rewards = []
        self.R_day = 0

        input_length = input_length*window_size+scenario-200
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(input_length,),dtype=self.dtype)
        
        # print(self.reduced.shape,self.observation_space.shape)
        if verbose:
            print(f'LOBenv_mid{self.__id} initialization complete.')

    def reset(self):

        self.action_list = []

        self.position_list = [(0,0,0)]
        self.position = 0
        self.position_price = None

        self.R_day = 0
        self.rewards = []
        
        mini_LOB_volumes = self.LOB_volumes.iloc[:self.window_size]
        mini_LOB_prices = self.__LOB_prices.iloc[:self.window_size]
        mini_LOB_mid_prices = self.LOB_mid_prices.iloc[:self.window_size]
        
        obs = self.reduced[:self.window_size].ravel() if self.use_pca else mini_LOB_volumes.to_numpy().ravel()
        if self.scenario > 201:
            obs = np.concatenate([obs,[0]],dtype=self.dtype)
            
        if self.scenario > 202:
            spread = mini_LOB_prices["ask1"].iloc[-1] - mini_LOB_prices["bid1"].iloc[-1]
            spread = round(spread,5)
            obs = np.concatenate([obs,[spread]],dtype=self.dtype)
        
        obs = np.concatenate([obs,[self.position - 1]],dtype=self.dtype)

        if self.prices_enabled and not self.use_pca:
            obs = np.concatenate([mini_LOB_mid_prices.to_numpy().ravel(),obs],dtype=self.dtype)

        
        if self.using_reservoir:
            obs = np.concatenate([self.reservoir_sample[:,0],obs])

        # obs = (prices; volumes; mark_to_market; spread; current_position)
        return obs#.astype(np.float64)
        

    def step(self, action):
        assert action in [0,1,2,3]
        
        done = False
        reward = 0
        
        self.action_list.append(action)
        no_of_actions = len(self.action_list)
        
        # current prices
        # mini_LOB_prices = self.LOB_prices.iloc[no_of_actions-1:no_of_actions-1+self.window_size]
        mini_LOB_mid_prices = self.__LOB_mid_prices.iloc[no_of_actions-1:no_of_actions-1+self.window_size]
        
        # next volumes
        mini_LOB_volumes = self.LOB_volumes.iloc[no_of_actions:no_of_actions+self.window_size]
        obs = self.reduced[no_of_actions:no_of_actions+self.window_size].ravel() if self.use_pca else mini_LOB_volumes.to_numpy().ravel()
        
        # No pos
        if self.position == 0:
            if action == 0:
                #short position
                self.position = 2
                self.position_price = mini_LOB_mid_prices.iloc[-1]
                self.position_list.append((no_of_actions,self.position,self.position_price))
                if self.spread:
                    self.position_price-=self.spread
            elif action == 2:
                #long position
                self.position = 1
                self.position_price = mini_LOB_mid_prices.iloc[-1]
                self.position_list.append((no_of_actions,self.position,self.position_price))
                if self.spread:
                    self.position_price+=self.spread
                
        # Long
        elif self.position == 1:
            if action == 0:
                #close long position
                reward = mini_LOB_mid_prices.iloc[-1] - self.position_price
                reward = round(reward,5)
                self.R_day += reward
                self.rewards.append(reward)
                self.position = 0
                self.position_price = None
                self.position_list.append((no_of_actions,self.position, reward))
                
        # Short
        elif self.position == 2:
            if action == 2:
                #close short position
                reward = self.position_price - mini_LOB_mid_prices.iloc[-1]
                reward = round(reward,5)
                self.R_day += reward
                self.rewards.append(reward)
                self.position = 0
                self.position_price = None
                self.position_list.append((no_of_actions,self.position, reward))
            

        # daily stop loss
        if action == 3 and self.R_day < 0:
            # if in Short
            if self.position == 2:
                reward = self.position_price - mini_LOB_mid_prices.iloc[-1]
                reward = round(reward,5)
                self.R_day += reward
                self.rewards.append(reward)
            # if in Long
            elif self.position == 1:
                reward = mini_LOB_mid_prices.iloc[-1] - self.position_price
                reward = round(reward,5)
                self.R_day += reward
                self.rewards.append(reward)
                
            self.position = 0
            self.position_price = None
            done = True
            self.position_list.append("done!")
                
        if self.scenario > 201:
            if self.position == 0:
                obs = np.concatenate([obs,[0]],dtype=self.dtype)
            elif self.position == 1:
                mark_to_market = self.__LOB_mid_prices[no_of_actions:no_of_actions+self.window_size]
                mark_to_market = mark_to_market.iloc[-1] - self.position_price
                mark_to_market = round(mark_to_market,5)
                obs = np.concatenate([obs,[mark_to_market]],dtype=self.dtype)
            elif self.position == 2:
                mark_to_market = self.__LOB_mid_prices[no_of_actions:no_of_actions+self.window_size]
                mark_to_market = self.position_price - mark_to_market.iloc[-1]
                mark_to_market = round(mark_to_market,5)
                obs = np.concatenate([obs,[mark_to_market]],dtype=self.dtype)
            else:
                raise Exception(f"Something is terribly wrong! self.position={self.position}")
                
                
        if self.scenario > 202:
            spread = self.__LOB_prices.iloc[no_of_actions:no_of_actions+self.window_size]["ask1"].iloc[-1] - self.__LOB_prices.iloc[no_of_actions:no_of_actions+self.window_size]["bid1"].iloc[-1]
            spread = round(spread,5)
            obs = np.concatenate([obs,[spread]],dtype=self.dtype)
                    
        
        obs = np.concatenate([obs,[self.position - 1]],dtype=self.dtype)
        
        # to add prices to observations
        if self.prices_enabled and not self.use_pca:
            # mini_LOB_prices = self.LOB_prices.iloc[no_of_actions:no_of_actions+self.window_size]
            mini_LOB_mid_prices = self.LOB_mid_prices.iloc[no_of_actions:no_of_actions+self.window_size]
            obs = np.concatenate([mini_LOB_mid_prices.to_numpy().ravel(),obs],dtype=self.dtype)
        
        if len(self.action_list) == self.day_ticksize - self.window_size:
            done = True
            self.position_list.append("done!")
            
        
        info = {}

        if self.using_reservoir:
            obs=np.concatenate([self.reservoir_sample[:,no_of_actions],obs])
        
        # obs = (prices; volumes; mark_to_market; spread; current_position)
        return obs, reward, done, info
        return obs.astype(np.float64), reward, done, info

    def render(self):
        pass

    def close(self):
        pass