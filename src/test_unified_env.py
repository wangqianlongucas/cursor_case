#!/usr/bin/env python3
"""
Test script for the unified TSP environment
"""

import numpy as np
import torch
from unified_env import UnifiedTSPEnvironment

def test_simple_mode():
    """Test the unified environment in simple mode (for Q-learning)."""
    print("Testing Unified Environment - Simple Mode")
    print("="*50)
    
    env = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='simple')
    
    print(f"Number of cities: {env.num_cities}")
    print(f"State mode: {env.state_mode}")
    print(f"State size: {env.get_state_size()}")
    
    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    print(f"Current city: {env.current_city}")
    print(f"Visited cities: {env.visited_cities}")
    
    # Test a few steps
    print("\nTesting steps:")
    for step in range(3):
        valid_actions = env.get_valid_actions()
        print(f"Step {step + 1}: Valid actions: {valid_actions}")
        
        if valid_actions:
            action = valid_actions[0]  # Take first valid action
            next_state, reward, done, info = env.step(action)
            print(f"  Action: {action}, Reward: {reward:.2f}, Done: {done}")
            print(f"  Next state: {next_state}")
            print(f"  Visited cities: {env.visited_cities}")
            
            if done:
                break
    
    print(f"Final distance: {env.total_distance:.2f}")
    print()

def test_enhanced_mode():
    """Test the unified environment in enhanced mode (for DQN/PPO)."""
    print("Testing Unified Environment - Enhanced Mode")
    print("="*50)
    
    env = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='enhanced')
    
    print(f"Number of cities: {env.num_cities}")
    print(f"State mode: {env.state_mode}")
    print(f"State size: {env.get_state_size()}")
    
    # Test reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state type: {type(state)}")
    print(f"Current city: {env.current_city}")
    print(f"Visited cities: {env.visited_cities}")
    
    # Test action mask
    action_mask = env.get_action_mask()
    print(f"Action mask: {action_mask}")
    print(f"Valid actions: {env.get_valid_actions()}")
    
    # Test a few steps
    print("\nTesting steps:")
    for step in range(3):
        valid_actions = env.get_valid_actions()
        action_mask = env.get_action_mask()
        print(f"Step {step + 1}: Valid actions: {valid_actions}")
        print(f"  Action mask: {action_mask}")
        
        if valid_actions:
            action = valid_actions[0]  # Take first valid action
            next_state, reward, done, info = env.step(action)
            print(f"  Action: {action}, Reward: {reward:.2f}, Done: {done}")
            print(f"  Next state shape: {next_state.shape}")
            print(f"  Visited cities: {env.visited_cities}")
            
            if done:
                break
    
    print(f"Final distance: {env.total_distance:.2f}")
    print()

def test_compatibility():
    """Test compatibility between modes."""
    print("Testing Mode Compatibility")
    print("="*50)
    
    # Create environments with same seed
    env_simple = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='simple')
    env_enhanced = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='enhanced')
    
    # Check if city positions are the same
    cities_match = np.allclose(env_simple.cities, env_enhanced.cities)
    print(f"City positions match: {cities_match}")
    
    # Check if distance matrices are the same
    distances_match = np.allclose(env_simple.distance_matrix, env_enhanced.distance_matrix)
    print(f"Distance matrices match: {distances_match}")
    
    # Test same sequence of actions
    env_simple.reset()
    env_enhanced.reset()
    
    actions = [1, 2, 3, 4]  # Visit cities in order
    
    simple_distances = []
    enhanced_distances = []
    
    for action in actions:
        # Simple mode
        _, reward_simple, done_simple, info_simple = env_simple.step(action)
        if info_simple["valid"]:
            simple_distances.append(env_simple.total_distance)
        
        # Enhanced mode
        _, reward_enhanced, done_enhanced, info_enhanced = env_enhanced.step(action)
        if info_enhanced["valid"]:
            enhanced_distances.append(env_enhanced.total_distance)
    
    print(f"Simple mode distances: {simple_distances}")
    print(f"Enhanced mode distances: {enhanced_distances}")
    print(f"Distance progression matches: {np.allclose(simple_distances, enhanced_distances)}")
    print()

def test_ppo_compatibility():
    """Test PPO-specific features."""
    print("Testing PPO Compatibility")
    print("="*50)
    
    env = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='enhanced')
    
    # Test state tensor conversion
    state = env.reset()
    print(f"State type: {type(state)}")
    print(f"State is tensor: {torch.is_tensor(state)}")
    
    # Test action mask
    action_mask = env.get_action_mask()
    print(f"Action mask type: {type(action_mask)}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Action mask sum (should equal valid actions): {action_mask.sum()}")
    print(f"Number of valid actions: {len(env.get_valid_actions())}")
    
    # Test multiple resets
    print("\nTesting multiple resets:")
    for i in range(3):
        state = env.reset()
        print(f"Reset {i+1}: Current city: {env.current_city}, State shape: {state.shape}")
    
    print()

def run_all_tests():
    """Run all tests."""
    print("Running Unified Environment Tests")
    print("="*60)
    print()
    
    test_simple_mode()
    test_enhanced_mode()
    test_compatibility()
    test_ppo_compatibility()
    
    print("All tests completed successfully! ✅")

if __name__ == "__main__":
    run_all_tests() 