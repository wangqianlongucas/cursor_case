import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from dqn_env import DQNTSPEnvironment
from dqn_agent import DQNAgent

def train_dqn(num_episodes=5000, num_cities=10, seed=42, save_model=True, model_dir="models"):
    """
    Train DQN agent on TSP environment
    
    Args:
        num_episodes (int): Number of training episodes
        num_cities (int): Number of cities in TSP
        seed (int): Random seed
        save_model (bool): Whether to save the trained model
        model_dir (str): Directory to save models
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = DQNTSPEnvironment(num_cities=num_cities, seed=seed)
    state_size = env.get_state_size()
    action_size = num_cities
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    )
    
    # Training metrics
    episode_rewards = []
    episode_distances = []
    episode_losses = []
    best_distance = float('inf')
    best_path = None
    best_episode = 0
    
    # Create model directory
    if save_model and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("Starting DQN training...")
    
    for episode in tqdm(range(num_episodes), desc="Training DQN"):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < num_cities * 2:  # Prevent infinite loops
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)
            
            if action is None:
                break
            
            next_state, reward, done, info = env.step(action)
            
            if info["valid"]:
                # Store experience in replay buffer
                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                
                # Train the agent
                agent.replay()
            else:
                # Invalid action taken, end episode
                break
            
            step_count += 1
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_distances.append(env.total_distance)
        
        # Track best solution
        if env.total_distance < best_distance and len(env.visited_cities) == num_cities:
            best_distance = env.total_distance
            best_path = env.visited_cities.copy()
            best_episode = episode
        
        # Record training statistics
        stats = agent.get_training_stats()
        episode_losses.append(stats['avg_loss'])
        
        # Print progress
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-500:])
            avg_distance = np.mean(episode_distances[-500:])
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward (last 500): {avg_reward:.2f}")
            print(f"Average Distance (last 500): {avg_distance:.2f}")
            print(f"Best Distance: {best_distance:.2f} (Episode {best_episode + 1})")
            print(f"Epsilon: {stats['epsilon']:.3f}")
            print(f"Buffer Size: {stats['buffer_size']}")
            print(f"Average Loss: {stats['avg_loss']:.4f}")
    
    # Save the trained model
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"dqn_tsp_{num_cities}cities_{timestamp}.pth")
        agent.save_model(model_path)
        print(f"Model saved to: {model_path}")
    
    return {
        'agent': agent,
        'env': env,
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'episode_losses': episode_losses,
        'best_distance': best_distance,
        'best_path': best_path,
        'best_episode': best_episode
    }

def plot_dqn_results(results):
    """Plot comprehensive DQN training results."""
    episode_rewards = results['episode_rewards']
    episode_distances = results['episode_distances']
    episode_losses = results['episode_losses']
    best_distance = results['best_distance']
    best_path = results['best_path']
    env = results['env']
    
    fig = plt.figure(figsize=(20, 12))
    
    # Episode rewards
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(episode_rewards, alpha=0.7)
    window_size = 100
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode distances
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(episode_distances, alpha=0.7)
    if len(episode_distances) >= window_size:
        moving_avg = np.convolve(episode_distances, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(episode_distances)), moving_avg, 'r-', linewidth=2, label='Moving Average')
    ax2.axhline(y=best_distance, color='g', linestyle='--', linewidth=2, label=f'Best: {best_distance:.2f}')
    ax2.set_title('Episode Distances')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Distance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training losses
    ax3 = plt.subplot(2, 4, 3)
    valid_losses = [loss for loss in episode_losses if loss > 0]
    if valid_losses:
        ax3.plot(valid_losses, alpha=0.7)
        if len(valid_losses) >= 50:
            moving_avg = np.convolve(valid_losses, np.ones(50)/50, mode='valid')
            ax3.plot(range(49, len(valid_losses)), moving_avg, 'r-', linewidth=2, label='Moving Average')
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('MSE Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distance distribution
    ax4 = plt.subplot(2, 4, 4)
    valid_distances = [d for d in episode_distances if d > 0]
    if valid_distances:
        ax4.hist(valid_distances, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(best_distance, color='red', linestyle='--', linewidth=2, label=f'Best: {best_distance:.2f}')
        ax4.axvline(np.mean(valid_distances), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(valid_distances):.2f}')
    ax4.set_title('Distance Distribution')
    ax4.set_xlabel('Total Distance')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Best solution visualization
    ax5 = plt.subplot(2, 4, (5, 8))
    ax5.scatter(env.cities[:, 0], env.cities[:, 1], c='red', s=200, zorder=3, edgecolors='black', linewidth=2)
    
    # Add city labels
    for i, (x, y) in enumerate(env.cities):
        ax5.annotate(str(i), (x, y), xytext=(0, 0), textcoords='offset points', 
                    fontsize=14, ha='center', va='center', weight='bold', color='white')
    
    # Plot best path
    if best_path and len(best_path) == env.num_cities:
        # Plot path segments
        for i in range(len(best_path) - 1):
            city1 = best_path[i]
            city2 = best_path[i + 1]
            ax5.plot([env.cities[city1, 0], env.cities[city2, 0]],
                    [env.cities[city1, 1], env.cities[city2, 1]], 'b-', linewidth=4, alpha=0.8)
            
            # Add direction arrows
            mid_x = (env.cities[city1, 0] + env.cities[city2, 0]) / 2
            mid_y = (env.cities[city1, 1] + env.cities[city2, 1]) / 2
            dx = env.cities[city2, 0] - env.cities[city1, 0]
            dy = env.cities[city2, 1] - env.cities[city1, 1]
            ax5.arrow(mid_x, mid_y, dx*0.15, dy*0.15, head_width=3, head_length=3, 
                     fc='blue', ec='blue', alpha=0.8, linewidth=2)
        
        # Return to start
        last_city = best_path[-1]
        ax5.plot([env.cities[last_city, 0], env.cities[0, 0]],
                [env.cities[last_city, 1], env.cities[0, 1]], 'b--', linewidth=4, alpha=0.8)
        
        # Highlight start city
        ax5.scatter(env.cities[0, 0], env.cities[0, 1], c='green', s=300, zorder=4, 
                   marker='*', edgecolors='black', linewidth=2, label='Start')
    
    ax5.set_title(f'Best Solution Found\nDistance: {best_distance:.2f} | Episode: {results["best_episode"] + 1}', 
                 fontsize=16, weight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=12)
    ax5.set_xlabel('X Coordinate', fontsize=12)
    ax5.set_ylabel('Y Coordinate', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"DQN TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Best Distance Found: {best_distance:.2f}")
    print(f"Best Episode: {results['best_episode'] + 1}")
    print(f"Best Path: {best_path}")
    print(f"Final Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final Average Distance (last 100): {np.mean(episode_distances[-100:]):.2f}")
    if valid_losses:
        print(f"Final Average Loss (last 100): {np.mean(valid_losses[-100:]):.4f}")
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"{'='*60}")

def compare_with_random_baseline(env, num_tests=1000):
    """Compare DQN performance with random baseline."""
    print("\nComparing with random baseline...")
    
    random_distances = []
    for _ in tqdm(range(num_tests), desc="Random baseline"):
        env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = np.random.choice(valid_actions)
            _, _, done, _ = env.step(action)
        
        if len(env.visited_cities) == env.num_cities:
            random_distances.append(env.total_distance)
    
    if random_distances:
        avg_random = np.mean(random_distances)
        min_random = np.min(random_distances)
        print(f"Random Policy - Average: {avg_random:.2f}, Best: {min_random:.2f}")
        return avg_random, min_random
    else:
        print("Random policy failed to complete any tours")
        return None, None

if __name__ == "__main__":
    # Training configuration
    config = {
        'num_episodes': 5000,
        'num_cities': 10,
        'seed': 42
    }
    
    print("Starting DQN TSP Training...")
    print(f"Configuration: {config}")
    
    # Train DQN
    results = train_dqn(**config)
    
    # Plot results
    plot_dqn_results(results)
    
    # Compare with random baseline
    avg_random, min_random = compare_with_random_baseline(results['env'])
    
    if avg_random:
        improvement = avg_random / results['best_distance']
        print(f"\nDQN vs Random Policy:")
        print(f"Improvement Factor: {improvement:.2f}x")
        print(f"DQN Best: {results['best_distance']:.2f}")
        print(f"Random Average: {avg_random:.2f}")
        print(f"Random Best: {min_random:.2f}") 