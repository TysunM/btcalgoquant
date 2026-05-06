import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from alpha_gen import generate_alpha_signals

class BtcSwingEnv(gym.Env):
    """
    Custom Environment for the Reinforcement Learning Agent.
    It simulates trading BTC over the historical data provided.
    """
    def __init__(self, df, initial_balance=100000):
        super(BtcSwingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_drawdown_limit = 0.20 # The strict 20% risk rule
        
        # Action Space: Continuous value between 0 and 1.
        # Represents the percentage of our portfolio to allocate to BTC right now.
        # 0 = All Cash, 1 = Fully Invested.
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space: What the agent "sees" at each step.
        # We pass 11 features: Open, High, Low, Close, Vol, 3 EMAs, MACD, MACD_hist, RSI
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth = self.initial_balance
        self.peak_net_worth = self.initial_balance
        
        return self._get_observation(), {}

    def _get_observation(self):
        # The agent looks at the current row's indicators
        features = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 
                    'EMA_10', 'EMA_50', 'EMA_200', 'MACD', 'MACD_hist', 'RSI_14']
        obs = self.df.loc[self.current_step, features].values.astype(np.float32)
        
        # In a production environment, we would also append the LSTM's probability score here
        return obs

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close_price']
        
        # Execute Action: Target portfolio allocation
        target_allocation_pct = action[0]
        target_btc_value = self.net_worth * target_allocation_pct
        
        # Adjust position
        current_btc_value = self.btc_held * current_price
        diff_value = target_btc_value - current_btc_value
        
        if diff_value > 0: # Buy
            if self.balance >= diff_value:
                self.btc_held += diff_value / current_price
                self.balance -= diff_value
        elif diff_value < 0: # Sell
            sell_amount = min(abs(diff_value) / current_price, self.btc_held)
            self.btc_held -= sell_amount
            self.balance += sell_amount * current_price

        # Step forward
        self.current_step += 1
        
        # Calculate new net worth
        if self.current_step >= len(self.df):
            done = True
            new_price = current_price
        else:
            done = False
            new_price = self.df.loc[self.current_step, 'close_price']
            
        self.net_worth = self.balance + (self.btc_held * new_price)
        
        # Track drawdown
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        
        # Reward Function Logic
        # Reward is the change in net worth, heavily penalized if drawdown nears 20%
        reward = self.net_worth - self.initial_balance
        
        if drawdown >= self.max_drawdown_limit:
            reward = -1000000  # Massive penalty for violating risk rule
            done = True # Terminate episode early
            
        info = {'net_worth': self.net_worth, 'drawdown': drawdown}
        
        # Truncated is a required boolean for newer gymnasium versions
        truncated = False 
        
        return self._get_observation(), reward, done, truncated, info

def train_rl_agent():
    print("Loading data for RL training...")
    df = generate_alpha_signals('1d')
    
    if df is None or len(df) < 100:
        print("Insufficient data. Run ingestion and alpha scripts first.")
        return
    
    print("Initializing environment...")
    # Wrap environment for Stable Baselines
    env = DummyVecEnv([lambda: BtcSwingEnv(df)])
    
    # Initialize PPO Agent
    # MlpPolicy is a standard neural network. We use a high learning rate to start.
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
    
    print("Beginning RL Agent Training (Playing the market)...")
    # Train the agent over 100,000 steps
    # Increase timesteps for better convergence later
    model.learn(total_timesteps=100000)
    
    model.save("ppo_btc_risk_manager")
    print("Training complete. Agent saved as ppo_btc_risk_manager.zip")

if __name__ == "__main__":
    train_rl_agent()
