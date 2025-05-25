import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
import time

from unified_env import UnifiedTSPEnvironment
from ppo_agent import PPOAgent

def train_ppo(num_episodes=10000, num_cities=10, seed=42, save_model=True, 
              model_dir="models", update_frequency=2048):
    """
    Train PPO agent on TSP environment
    
    Args:
        num_episodes (int): Number of training episodes
        num_cities (int): Number of cities in TSP
        seed (int): Random seed
        save_model (bool): Whether to save the trained model
        model_dir (str): Directory to save models
        update_frequency (int): Update policy every N steps
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='enhanced')
    state_size = env.get_state_size()
    action_size = num_cities
    
    print(f"PPO TSP Training Configuration:")
    print(f"  Cities: {num_cities}")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Update frequency: {update_frequency}")
    
    # Create PPO agent
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        buffer_size=update_frequency
    )
    
    # Training metrics
    episode_rewards = []
    episode_distances = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropy_losses = []
    
    best_distance = float('inf')
    best_path = None
    best_episode = 0
    
    # Create model directory
    if save_model and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print("\nStarting PPO training...")
    start_time = time.time()
    
    step_count = 0
    
    for episode in tqdm(range(num_episodes), desc="Training PPO"):
        state = env.reset()
        total_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < num_cities * 2:  # Prevent infinite loops
            action_mask = env.get_action_mask()
            action = agent.get_action(state, action_mask)
            
            next_state, reward, done, info = env.step(action)
            
            if info["valid"]:
                # Store transition
                agent.store_transition(state, action, reward, done, action_mask)
                total_reward += reward
                state = next_state
                step_count += 1
                episode_length += 1
                
                # Update policy if buffer is full
                if step_count % update_frequency == 0:
                    # Get next state value for GAE computation
                    if not done:
                        next_action_mask = env.get_action_mask()
                        with torch.no_grad():
                            _, _, next_value = agent.actor_critic.get_action_and_value(
                                torch.FloatTensor(next_state).unsqueeze(0).to(agent.device),
                                torch.FloatTensor(next_action_mask).unsqueeze(0).to(agent.device)
                            )
                        next_value = next_value.cpu().item()
                    else:
                        next_value = 0
                    
                    agent.update(next_value)
                    
                    # Record training statistics
                    stats = agent.get_training_stats()
                    policy_losses.append(stats['policy_loss'])
                    value_losses.append(stats['value_loss'])
                    entropy_losses.append(stats['entropy_loss'])
            else:
                # Invalid action taken, end episode
                break
        
        # Record episode metrics
        episode_rewards.append(total_reward)
        episode_distances.append(env.total_distance)
        episode_lengths.append(episode_length)
        
        # Track best solution
        if env.total_distance < best_distance and len(env.visited_cities) == num_cities:
            best_distance = env.total_distance
            best_path = env.visited_cities.copy()
            best_episode = episode
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            avg_distance = np.mean(episode_distances[-1000:])
            avg_length = np.mean(episode_lengths[-1000:])
            
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward (last 1000): {avg_reward:.2f}")
            print(f"Average Distance (last 1000): {avg_distance:.2f}")
            print(f"Average Episode Length (last 1000): {avg_length:.1f}")
            print(f"Best Distance: {best_distance:.2f} (Episode {best_episode + 1})")
            
            if policy_losses:
                print(f"Recent Policy Loss: {policy_losses[-1]:.4f}")
                print(f"Recent Value Loss: {value_losses[-1]:.4f}")
                print(f"Recent Entropy Loss: {entropy_losses[-1]:.4f}")
    
    training_time = time.time() - start_time
    
    # Save the trained model
    if save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"ppo_tsp_{num_cities}cities_{timestamp}.pth")
        agent.save_model(model_path)
        print(f"Model saved to: {model_path}")
    
    return {
        'agent': agent,
        'env': env,
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'episode_lengths': episode_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropy_losses': entropy_losses,
        'best_distance': best_distance,
        'best_path': best_path,
        'best_episode': best_episode,
        'training_time': training_time
    }

def plot_ppo_results(results):
    """Plot comprehensive PPO training results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    episodes = range(len(results['episode_rewards']))
    
    # Episode rewards
    axes[0, 0].plot(episodes, results['episode_rewards'], alpha=0.6)
    axes[0, 0].plot(episodes, smooth_curve(results['episode_rewards'], 100), 'r-', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode distances
    axes[0, 1].plot(episodes, results['episode_distances'], alpha=0.6)
    axes[0, 1].plot(episodes, smooth_curve(results['episode_distances'], 100), 'r-', linewidth=2)
    axes[0, 1].axhline(y=results['best_distance'], color='g', linestyle='--', 
                       label=f'Best: {results["best_distance"]:.2f}')
    axes[0, 1].set_title('Episode Distances')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Distance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 2].plot(episodes, results['episode_lengths'], alpha=0.6)
    axes[0, 2].plot(episodes, smooth_curve(results['episode_lengths'], 100), 'r-', linewidth=2)
    axes[0, 2].set_title('Episode Lengths')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps per Episode')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Training losses
    if results['policy_losses']:
        loss_episodes = range(len(results['policy_losses']))
        axes[1, 0].plot(loss_episodes, results['policy_losses'], label='Policy Loss')
        axes[1, 0].plot(loss_episodes, results['value_losses'], label='Value Loss')
        axes[1, 0].plot(loss_episodes, results['entropy_losses'], label='Entropy Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Performance distribution
    axes[1, 1].hist(results['episode_distances'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=results['best_distance'], color='r', linestyle='--', 
                       label=f'Best: {results["best_distance"]:.2f}')
    axes[1, 1].axvline(x=np.mean(results['episode_distances']), color='g', linestyle='--', 
                       label=f'Mean: {np.mean(results["episode_distances"]):.2f}')
    axes[1, 1].set_title('Distance Distribution')
    axes[1, 1].set_xlabel('Total Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Best solution visualization
    env = results['env']
    env.visited_cities = results['best_path']
    env.current_city = results['best_path'][-1]
    env.total_distance = results['best_distance']
    
    axes[1, 2].scatter(env.cities[:, 0], env.cities[:, 1], c='red', s=150, zorder=3)
    
    # Add city labels
    for i, (x, y) in enumerate(env.cities):
        color = 'blue' if i in env.visited_cities else 'black'
        weight = 'bold' if i == env.current_city else 'normal'
        axes[1, 2].annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, color=color, weight=weight)
    
    # Plot the best path
    for i in range(len(env.visited_cities) - 1):
        city1 = env.visited_cities[i]
        city2 = env.visited_cities[i + 1]
        axes[1, 2].plot([env.cities[city1, 0], env.cities[city2, 0]],
                       [env.cities[city1, 1], env.cities[city2, 1]], 'b-', linewidth=2)
    
    # Return to start
    if len(env.visited_cities) == env.num_cities:
        last_city = env.visited_cities[-1]
        axes[1, 2].plot([env.cities[last_city, 0], env.cities[0, 0]],
                       [env.cities[last_city, 1], env.cities[0, 1]], 'b--', linewidth=2)
    
    axes[1, 2].set_title(f'Best Solution (Distance: {results["best_distance"]:.2f})')
    axes[1, 2].set_xlabel('X Coordinate')
    axes[1, 2].set_ylabel('Y Coordinate')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("PPO TRAINING SUMMARY")
    print("="*60)
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Total Episodes: {len(results['episode_rewards'])}")
    print(f"Best Distance: {results['best_distance']:.2f}")
    print(f"Best Path: {results['best_path']}")
    print(f"Best Episode: {results['best_episode'] + 1}")
    print(f"Final Average Distance (last 1000): {np.mean(results['episode_distances'][-1000:]):.2f}")
    print(f"Final Average Reward (last 1000): {np.mean(results['episode_rewards'][-1000:]):.2f}")
    print(f"Final Average Episode Length (last 1000): {np.mean(results['episode_lengths'][-1000:]):.1f}")

def smooth_curve(data, window_size):
    """Smooth curve using moving average."""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed

def compare_with_random_baseline(env, num_tests=1000):
    """Compare PPO performance with random baseline."""
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

def test_trained_agent(agent, env, num_tests=100):
    """Test the trained PPO agent."""
    print(f"\nTesting trained PPO agent ({num_tests} episodes)...")
    
    test_distances = []
    test_rewards = []
    successful_tours = 0
    
    for _ in tqdm(range(num_tests), desc="Testing"):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.num_cities * 2:
            action_mask = env.get_action_mask()
            action = agent.get_action(state, action_mask)
            state, reward, done, info = env.step(action)
            
            if info["valid"]:
                total_reward += reward
                steps += 1
            else:
                break
        
        if len(env.visited_cities) == env.num_cities:
            test_distances.append(env.total_distance)
            test_rewards.append(total_reward)
            successful_tours += 1
    
    if test_distances:
        print(f"Test Results:")
        print(f"  Successful tours: {successful_tours}/{num_tests}")
        print(f"  Average distance: {np.mean(test_distances):.2f}")
        print(f"  Best distance: {np.min(test_distances):.2f}")
        print(f"  Worst distance: {np.max(test_distances):.2f}")
        print(f"  Average reward: {np.mean(test_rewards):.2f}")
        
        return {
            'distances': test_distances,
            'rewards': test_rewards,
            'success_rate': successful_tours / num_tests
        }
    else:
        print("No successful tours in testing!")
        return None

if __name__ == "__main__":
    # Training configuration
    config = {
        'num_episodes': 10000,
        'num_cities': 10,
        'seed': 42,
        'update_frequency': 2048
    }
    
    print("Starting PPO TSP Training...")
    print(f"Configuration: {config}")
    
    # Train PPO
    results = train_ppo(**config)
    
    # Plot results
    plot_ppo_results(results)
    
    # Test trained agent
    test_results = test_trained_agent(results['agent'], results['env'])
    
    # Compare with random baseline
    avg_random, min_random = compare_with_random_baseline(results['env'])
    
    if avg_random and test_results:
        improvement = avg_random / np.mean(test_results['distances'])
        print(f"\nPPO vs Random Policy:")
        print(f"Improvement Factor: {improvement:.2f}x")
        print(f"PPO Average: {np.mean(test_results['distances']):.2f}")
        print(f"Random Average: {avg_random:.2f}")
        print(f"PPO Best: {np.min(test_results['distances']):.2f}")
        print(f"Random Best: {min_random:.2f}") 