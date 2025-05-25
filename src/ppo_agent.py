import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[512, 256, 128]):
        """
        Actor-Critic network for PPO
        
        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            hidden_sizes (list): Hidden layer sizes
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction layers
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[-1], action_size)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action_mask=None):
        """
        Forward pass through the network
        
        Args:
            state (torch.Tensor): State tensor
            action_mask (torch.Tensor): Mask for valid actions (1 for valid, 0 for invalid)
            
        Returns:
            tuple: (action_logits, state_value)
        """
        shared_features = self.shared_layers(state)
        
        # Actor output
        action_logits = self.actor(shared_features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set logits of invalid actions to very negative value
            action_logits = action_logits + (action_mask - 1) * 1e8
        
        # Critic output
        state_value = self.critic(shared_features)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state, action_mask=None):
        """
        Get action and value for given state
        
        Args:
            state (torch.Tensor): State tensor
            action_mask (torch.Tensor): Mask for valid actions
            
        Returns:
            tuple: (action, log_prob, value)
        """
        action_logits, value = self.forward(state, action_mask)
        
        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions, action_masks=None):
        """
        Evaluate actions for given states
        
        Args:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions
            action_masks (torch.Tensor): Batch of action masks
            
        Returns:
            tuple: (log_probs, values, entropy)
        """
        action_logits, values = self.forward(states, action_masks)
        
        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        
        # Calculate log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, ppo_epochs=4, batch_size=64, buffer_size=2048):
        """
        PPO Agent for TSP
        
        Args:
            state_size (int): Size of state vector
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            clip_epsilon (float): PPO clipping parameter
            value_coef (float): Value loss coefficient
            entropy_coef (float): Entropy loss coefficient
            max_grad_norm (float): Maximum gradient norm for clipping
            ppo_epochs (int): Number of PPO update epochs
            batch_size (int): Batch size for training
            buffer_size (int): Size of experience buffer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")
        
        # Networks
        self.actor_critic = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.reset_buffer()
        
        # Training statistics
        self.training_stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy_loss': 0,
            'total_loss': 0,
            'explained_variance': 0,
            'clipfrac': 0,
            'approx_kl': 0
        }
    
    def reset_buffer(self):
        """Reset the experience buffer."""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'action_masks': []
        }
    
    def get_action(self, state, action_mask):
        """
        Get action for given state
        
        Args:
            state (torch.Tensor or np.array): Current state
            action_mask (np.array): Valid action mask
            
        Returns:
            int: Selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action_and_value(state, action_mask)
        
        # Store for training
        self.current_log_prob = log_prob.cpu().item()
        self.current_value = value.cpu().item()
        
        return action.cpu().item()
    
    def store_transition(self, state, action, reward, done, action_mask):
        """
        Store transition in buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            done: Whether episode is done
            action_mask: Valid action mask
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.FloatTensor(action_mask)
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(self.current_value)
        self.buffer['log_probs'].append(self.current_log_prob)
        self.buffer['dones'].append(done)
        self.buffer['action_masks'].append(action_mask)
    
    def compute_gae(self, next_value=0):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            next_value: Value of next state (0 if terminal)
            
        Returns:
            tuple: (advantages, returns)
        """
        rewards = self.buffer['rewards']
        values = self.buffer['values']
        dones = self.buffer['dones']
        
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value_i = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value_i = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value_i * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self, next_value=0):
        """
        Update the policy using PPO
        
        Args:
            next_value: Value of next state
        """
        if len(self.buffer['states']) < self.batch_size:
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.stack(self.buffer['states']).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        action_masks = torch.stack(self.buffer['action_masks']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                
                # Evaluate current policy
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions, batch_action_masks
                )
                
                # Calculate ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update statistics
                with torch.no_grad():
                    approx_kl = ((log_probs - batch_old_log_probs) ** 2).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    explained_var = 1 - F.mse_loss(values, batch_returns) / batch_returns.var()
                
                self.training_stats.update({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy_loss': entropy_loss.item(),
                    'total_loss': total_loss.item(),
                    'explained_variance': explained_var.item(),
                    'clipfrac': clipfrac.item(),
                    'approx_kl': approx_kl.item()
                })
        
        # Reset buffer
        self.reset_buffer()
    
    def get_training_stats(self):
        """Get training statistics."""
        return self.training_stats.copy()
    
    def save_model(self, filepath):
        """Save the model."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"Model loaded from {filepath}") 