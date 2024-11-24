import numpy as np
from scheduler import Scheduler
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.q_table = {}  # Using dictionary for sparse state representation
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_size = action_size

    def get_action(self, state):
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = str(state)
        next_state_key = str(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        if not done:
            next_max = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * next_max
        else:
            target = reward

        self.q_table[state_key][action] = (1 - self.lr) * self.q_table[state_key][action] + self.lr * target
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train():
    # Environment parameters
    n_jobs = 100
    max_time = 40
    max_resources = 30
    
    # Training parameters
    n_episodes = 1000
    
    # Initialize environment and agent
    env = Scheduler(n=n_jobs, mt=max_time, mr=max_resources)
    agent = QLearningAgent(state_size=None, action_size=max_time)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action + 1)  # +1 because actions are 1-based

            if done :
                print("Done!")
            
            # Let agent learn from experience
            agent.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if episode % 100 == 0:  # Render every 100th episode
                # env.render()
                print(f"Action: {action + 1}, Reward: {reward}")
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode: {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

if __name__ == "__main__":
    train()