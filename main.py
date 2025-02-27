import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
# Additional imports for data fetching

class BitcoinTradingEnv(gym.Env):
    """Custom Environment for Bitcoin trading using RL"""
    
    def __init__(self, data, initial_balance=10000, commission=0.001):
        """
        Initialize the Bitcoin trading environment
        
        Args:
            data: DataFrame with Bitcoin price data
            initial_balance: Starting USD balance
            commission: Trading fee as a decimal percentage
        """
        super(BitcoinTradingEnv, self).__init__()
        
        # Store data and parameters
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: technical indicators + account info
        # Adjust the shape based on your features
        feature_count = 10  # Example: price + 9 technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        # Initialize environment state
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        # Reset to beginning of data
        self.current_step = 0
        
        # Reset portfolio state
        self.balance = self.initial_balance
        self.btc_held = 0
        self.net_worth_history = [self.initial_balance]
        
        # Get initial observation
        return self._next_observation()
    
    def _next_observation(self):
        """Get current market observation"""
        # Get current frame of market data
        frame = self.data.iloc[self.current_step]
        
        # Features based on current price data
        features = [
            frame.close,
            frame.volume,
            # Add technical indicators here, e.g.:
            frame.close / frame.open - 1,  # Price change
            # More features like MA, RSI, MACD, etc.
        ]
        
        # Add portfolio information
        features.append(self.balance)
        features.append(self.btc_held)
        features.append(self.balance + self.btc_held * frame.close)  # Net worth
        
        return np.array(features)
    
    def step(self, action):
        """
        Execute one step in the environment
        
        Args:
            action: 0 (HOLD), 1 (BUY), 2 (SELL)
            
        Returns:
            observation, reward, done, info
        """
        # Get current price data
        current_price = self.data.iloc[self.current_step].close
        
        # Default values
        reward = 0
        done = False
        info = {}
        
        # Execute action
        if action == 1:  # BUY
            # Calculate max BTC that can be bought
            max_btc_to_buy = self.balance / current_price
            # Apply commission
            btc_bought = max_btc_to_buy * (1 - self.commission)
            
            self.btc_held += btc_bought
            self.balance = 0  # All-in strategy
            
        elif action == 2:  # SELL
            # Calculate USD from selling all BTC
            usd_from_sale = self.btc_held * current_price
            # Apply commission
            self.balance += usd_from_sale * (1 - self.commission)
            self.btc_held = 0
        
        # Calculate current net worth
        self.current_net_worth = self.balance + self.btc_held * current_price
        self.net_worth_history.append(self.current_net_worth)
        
        # Calculate reward - based on change in net worth
        reward = self.net_worth_history[-1] - self.net_worth_history[-2]
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            done = True
        
        # Prepare info dict
        info = {
            'step': self.current_step,
            'net_worth': self.current_net_worth,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'current_price': current_price
        }
        
        # Get new observation
        obs = self._next_observation()
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment state"""
        current_price = self.data.iloc[self.current_step].close
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"BTC Held: {self.btc_held:.8f}")
        print(f"Net Worth: ${self.current_net_worth:.2f}")
        print("----------------------------------------")


def get_bitcoin_data(start_date, end_date, timeframe='1h'):
    """
    Fetch historical Bitcoin price data
    In a real implementation, this would use an API like ccxt, yfinance, etc.
    """
    # Placeholder - in real code, fetch from API
    df = pd.DataFrame({
        'open': np.random.random(1000) * 100 + 20000,
        'high': np.random.random(1000) * 100 + 20100,
        'low': np.random.random(1000) * 100 + 19900,
        'close': np.random.random(1000) * 100 + 20000,
        'volume': np.random.random(1000) * 1000
    })
    
    # Add technical indicators here
    # Example: add moving averages, RSI, MACD, etc.
    
    return df


def main():
    # Fetch and prepare data
    data = get_bitcoin_data(
        start_date='2020-01-01', 
        end_date='2023-01-01',
        timeframe='1h'
    )
    
    # Create environment
    env = BitcoinTradingEnv(data)
    
    # Define neural network structure
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                        net_arch=dict(pi=[128, 64, 64], vf=[128, 64, 64]))
    
    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    
    # Train the model
    model.learn(total_timesteps=100000)
    
    # Save the trained model
    model.save("bitcoin_trading_ppo")
    
    print("Training complete! Model saved as bitcoin_trading_ppo")
    
    # Optional: Test the model
    # test_model(model, env)


def test_model(model, env):
    """Test the trained model's performance"""
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Final net worth: ${env.current_net_worth:.2f}")
    print(f"Initial investment: ${env.initial_balance:.2f}")
    print(f"Return: {(env.current_net_worth / env.initial_balance - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()