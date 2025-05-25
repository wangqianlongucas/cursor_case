import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[512, 256, 128]):
        """
        Deep Q-Network architecture
        
        Args:
            state_size (int): Size of the state vector
            action_size (int): Number of possible actions
            hidden_sizes (list): Sizes of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        # Build the network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        """
        Experience replay buffer
        
        Args:
            capacity (int): Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 buffer_size=100000, batch_size=64, target_update_freq=1000):
        """
        DQN Agent with experience replay and target network
        
        Args:
            state_size (int): Size of the state vector
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network with same weights
        self.update_target_network()
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_step = 0
        self.losses = []
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, valid_actions):
        """
        Choose action using epsilon-greedy policy with valid action masking
        
        Args:
            state (torch.Tensor): Current state
            valid_actions (list): List of valid actions
            
        Returns:
            int: Selected action
        """
        if len(valid_actions) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get Q-values for all actions
        state = state.unsqueeze(0).to(self.device)
        q_values = self.q_network(state).cpu().data.numpy()[0]
        
        # Mask invalid actions
        masked_q_values = np.full(self.action_size, -np.inf)
        for action in valid_actions:
            masked_q_values[action] = q_values[action]
        
        return np.argmax(masked_q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(
            state.cpu().numpy(),
            action,
            reward,
            next_state.cpu().numpy(),
            done
        )
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Store loss for monitoring
        self.losses.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
    
    def get_training_stats(self):
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'buffer_size': len(self.replay_buffer)
        } 