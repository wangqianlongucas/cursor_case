import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tsp_env import TSPEnvironment
from q_learning import QLearningAgent

def train(num_episodes=2000, num_cities=10, seed=42):
    """
    Train the Q-learning agent on the TSP environment
    
    Args:
        num_episodes (int): Number of training episodes
        num_cities (int): Number of cities in the TSP
        seed (int): Random seed for reproducibility
    """
    # Initialize environment and agent
    env = TSPEnvironment(num_cities=num_cities, seed=seed)
    agent = QLearningAgent(num_cities=num_cities)
    
    # Lists to store metrics
    episode_rewards = []
    total_distances = []
    best_distance = float('inf')
    best_path = None
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, env.current_city, valid_actions)
            
            if action is None:
                break
                
            next_state, reward, done, info = env.step(action)
            
            if info["valid"]:
                next_valid_actions = env.get_valid_actions()
                agent.learn(state, env.visited_cities[-2] if len(env.visited_cities) > 1 else 0, 
                           action, reward, next_state, env.current_city, next_valid_actions, done)
                total_reward += reward
            
            state = next_state
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Store metrics and track best solution
        episode_rewards.append(total_reward)
        total_distances.append(env.total_distance)
        
        if env.total_distance < best_distance:
            best_distance = env.total_distance
            best_path = env.visited_cities.copy()
        
        # Print progress every 200 episodes
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_distance = np.mean(total_distances[-200:])
            print(f"\nEpisode {episode + 1}")
            print(f"Average Reward (last 200): {avg_reward:.2f}")
            print(f"Average Distance (last 200): {avg_distance:.2f}")
            print(f"Best Distance So Far: {best_distance:.2f}")
            print(f"Current Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards, total_distances, env, best_distance, best_path

def plot_final_results(episode_rewards, total_distances, env, best_distance, best_path):
    """Plot final training metrics and solution."""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episode_rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)
    
    # Plot total distances
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(total_distances)
    ax2.axhline(y=best_distance, color='r', linestyle='--', label=f'Best: {best_distance:.2f}')
    ax2.set_title("Total Distances")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot moving average of distances
    ax3 = plt.subplot(2, 3, 3)
    window_size = 100
    if len(total_distances) >= window_size:
        moving_avg = np.convolve(total_distances, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(range(window_size-1, len(total_distances)), moving_avg)
    ax3.set_title(f"Moving Average Distance (window={window_size})")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Distance")
    ax3.grid(True, alpha=0.3)
    
    # Plot final solution
    ax4 = plt.subplot(2, 3, (4, 6))
    ax4.scatter(env.cities[:, 0], env.cities[:, 1], c='red', s=100, zorder=3)
    
    # Add city labels
    for i, (x, y) in enumerate(env.cities):
        ax4.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot the best path
    if best_path:
        for i in range(len(best_path) - 1):
            city1 = best_path[i]
            city2 = best_path[i + 1]
            ax4.plot([env.cities[city1, 0], env.cities[city2, 0]],
                    [env.cities[city1, 1], env.cities[city2, 1]], 'b-', linewidth=2)
        
        # Plot return to start
        last_city = best_path[-1]
        ax4.plot([env.cities[last_city, 0], env.cities[0, 0]],
                [env.cities[last_city, 1], env.cities[0, 1]], 'b--', linewidth=2)
    
    ax4.set_title(f'Best Solution Found (Distance: {best_distance:.2f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining Summary:")
    print(f"Best distance found: {best_distance:.2f}")
    print(f"Best path: {best_path}")
    print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final average distance (last 100): {np.mean(total_distances[-100:]):.2f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Train the agent
    print("Starting training...")
    episode_rewards, total_distances, env, best_distance, best_path = train(num_episodes=2000, num_cities=10)
    
    # Plot final results
    plot_final_results(episode_rewards, total_distances, env, best_distance, best_path) 