import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, num_cities, learning_rate=0.3, discount_factor=0.9, epsilon=0.9, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Initialize Q-learning agent with improved hyperparameters
        
        Args:
            num_cities (int): Number of cities in the TSP
            learning_rate (float): Learning rate for Q-learning
            discount_factor (float): Discount factor for future rewards
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for epsilon
        """
        self.num_cities = num_cities
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table as a defaultdict to handle state-action pairs dynamically
        self.q_table = defaultdict(lambda: np.zeros(num_cities))
        
    def _state_to_key(self, state, current_city):
        """Convert state array and current city to hashable tuple for better state representation."""
        return (tuple(state), current_city)
    
    def get_action(self, state, current_city, valid_actions):
        """
        Get action using epsilon-greedy policy with improved state representation
        
        Args:
            state: Current state (visited cities)
            current_city (int): Current city position
            valid_actions (list): List of valid actions
            
        Returns:
            int: Selected action
        """
        if len(valid_actions) == 0:
            return None
            
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        state_key = self._state_to_key(state, current_city)
        q_values = self.q_table[state_key]
        
        # Select the action with highest Q-value among valid actions
        valid_q_values = {action: q_values[action] for action in valid_actions}
        return max(valid_q_values.items(), key=lambda x: x[1])[0]
    
    def learn(self, state, current_city, action, reward, next_state, next_current_city, next_valid_actions, done):
        """
        Update Q-values using Q-learning update rule with improved learning
        
        Args:
            state: Current state
            current_city (int): Current city
            action (int): Taken action
            reward (float): Received reward
            next_state: Next state
            next_current_city (int): Next current city
            next_valid_actions (list): Valid actions for next state
            done (bool): Whether episode is finished
        """
        state_key = self._state_to_key(state, current_city)
        next_state_key = self._state_to_key(next_state, next_current_city)
        
        # Get maximum Q-value for next state among valid actions
        if not done and len(next_valid_actions) > 0:
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values[action] for action in next_valid_actions)
        else:
            max_next_q = 0
        
        # Q-learning update rule
        current_q = self.q_table[state_key][action]
        target = reward + self.discount_factor * max_next_q
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate with improved decay strategy."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) 