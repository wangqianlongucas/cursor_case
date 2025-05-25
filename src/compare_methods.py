import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Import Q-learning components
from tsp_env import TSPEnvironment
from q_learning import QLearningAgent

# Import DQN components
from dqn_env import DQNTSPEnvironment
from dqn_agent import DQNAgent

def train_q_learning(num_episodes, num_cities, seed):
    """Train Q-learning agent and return results."""
    env = TSPEnvironment(num_cities=num_cities, seed=seed)
    agent = QLearningAgent(num_cities=num_cities)
    
    episode_rewards = []
    episode_distances = []
    best_distance = float('inf')
    best_path = None
    
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc="Training Q-Learning"):
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
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_distances.append(env.total_distance)
        
        if env.total_distance < best_distance and len(env.visited_cities) == num_cities:
            best_distance = env.total_distance
            best_path = env.visited_cities.copy()
    
    training_time = time.time() - start_time
    
    return {
        'method': 'Q-Learning',
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'best_distance': best_distance,
        'best_path': best_path,
        'training_time': training_time,
        'env': env
    }

def train_dqn(num_episodes, num_cities, seed):
    """Train DQN agent and return results."""
    env = DQNTSPEnvironment(num_cities=num_cities, seed=seed)
    state_size = env.get_state_size()
    agent = DQNAgent(
        state_size=state_size,
        action_size=num_cities,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    episode_rewards = []
    episode_distances = []
    best_distance = float('inf')
    best_path = None
    
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc="Training DQN"):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < num_cities * 2:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)
            
            if action is None:
                break
            
            next_state, reward, done, info = env.step(action)
            
            if info["valid"]:
                agent.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                agent.replay()
            else:
                break
            
            step_count += 1
        
        episode_rewards.append(total_reward)
        episode_distances.append(env.total_distance)
        
        if env.total_distance < best_distance and len(env.visited_cities) == num_cities:
            best_distance = env.total_distance
            best_path = env.visited_cities.copy()
    
    training_time = time.time() - start_time
    
    return {
        'method': 'DQN',
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'best_distance': best_distance,
        'best_path': best_path,
        'training_time': training_time,
        'env': env
    }

def compare_methods(num_episodes=3000, num_cities=10, seed=42):
    """Compare Q-learning and DQN on TSP."""
    print(f"Comparing Q-Learning vs DQN on {num_cities}-city TSP")
    print(f"Training episodes: {num_episodes}")
    print(f"Random seed: {seed}")
    print("="*60)
    
    # Train both methods
    q_results = train_q_learning(num_episodes, num_cities, seed)
    dqn_results = train_dqn(num_episodes, num_cities, seed)
    
    # Calculate random baseline
    print("\nCalculating random baseline...")
    env = TSPEnvironment(num_cities=num_cities, seed=seed)
    random_distances = []
    
    for _ in tqdm(range(1000), desc="Random baseline"):
        env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = np.random.choice(valid_actions)
            _, _, done, _ = env.step(action)
        
        if len(env.visited_cities) == num_cities:
            random_distances.append(env.total_distance)
    
    random_avg = np.mean(random_distances) if random_distances else float('inf')
    random_best = np.min(random_distances) if random_distances else float('inf')
    
    # Print comparison results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nQ-Learning:")
    print(f"  Best Distance: {q_results['best_distance']:.2f}")
    print(f"  Best Path: {q_results['best_path']}")
    print(f"  Final Avg Distance (last 100): {np.mean(q_results['episode_distances'][-100:]):.2f}")
    print(f"  Training Time: {q_results['training_time']:.2f} seconds")
    
    print(f"\nDQN:")
    print(f"  Best Distance: {dqn_results['best_distance']:.2f}")
    print(f"  Best Path: {dqn_results['best_path']}")
    print(f"  Final Avg Distance (last 100): {np.mean(dqn_results['episode_distances'][-100:]):.2f}")
    print(f"  Training Time: {dqn_results['training_time']:.2f} seconds")
    
    print(f"\nRandom Baseline:")
    print(f"  Average Distance: {random_avg:.2f}")
    print(f"  Best Distance: {random_best:.2f}")
    
    print(f"\nImprovement over Random:")
    print(f"  Q-Learning: {random_avg / q_results['best_distance']:.2f}x")
    print(f"  DQN: {random_avg / dqn_results['best_distance']:.2f}x")
    
    if q_results['best_distance'] < dqn_results['best_distance']:
        improvement = dqn_results['best_distance'] / q_results['best_distance']
        print(f"\nQ-Learning is {improvement:.2f}x better than DQN")
    else:
        improvement = q_results['best_distance'] / dqn_results['best_distance']
        print(f"\nDQN is {improvement:.2f}x better than Q-Learning")
    
    # Plot comparison
    plot_comparison(q_results, dqn_results, random_avg, random_best)
    
    return q_results, dqn_results, random_avg, random_best

def plot_comparison(q_results, dqn_results, random_avg, random_best):
    """Plot comparison between Q-learning and DQN."""
    fig = plt.figure(figsize=(20, 12))
    
    # Training curves - Distances
    ax1 = plt.subplot(2, 4, 1)
    window_size = 100
    
    # Q-Learning
    q_distances = q_results['episode_distances']
    ax1.plot(q_distances, alpha=0.3, color='blue', label='Q-Learning (raw)')
    if len(q_distances) >= window_size:
        q_moving_avg = np.convolve(q_distances, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(q_distances)), q_moving_avg, 'b-', linewidth=2, label='Q-Learning (avg)')
    
    # DQN
    dqn_distances = dqn_results['episode_distances']
    ax1.plot(dqn_distances, alpha=0.3, color='red', label='DQN (raw)')
    if len(dqn_distances) >= window_size:
        dqn_moving_avg = np.convolve(dqn_distances, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(dqn_distances)), dqn_moving_avg, 'r-', linewidth=2, label='DQN (avg)')
    
    # Baselines
    ax1.axhline(y=random_avg, color='gray', linestyle='--', label=f'Random Avg: {random_avg:.2f}')
    ax1.axhline(y=q_results['best_distance'], color='blue', linestyle=':', label=f'Q-Learning Best: {q_results["best_distance"]:.2f}')
    ax1.axhline(y=dqn_results['best_distance'], color='red', linestyle=':', label=f'DQN Best: {dqn_results["best_distance"]:.2f}')
    
    ax1.set_title('Training Progress - Distance')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training curves - Rewards
    ax2 = plt.subplot(2, 4, 2)
    
    # Q-Learning rewards
    q_rewards = q_results['episode_rewards']
    ax2.plot(q_rewards, alpha=0.3, color='blue')
    if len(q_rewards) >= window_size:
        q_reward_avg = np.convolve(q_rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(q_rewards)), q_reward_avg, 'b-', linewidth=2, label='Q-Learning')
    
    # DQN rewards
    dqn_rewards = dqn_results['episode_rewards']
    ax2.plot(dqn_rewards, alpha=0.3, color='red')
    if len(dqn_rewards) >= window_size:
        dqn_reward_avg = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(dqn_rewards)), dqn_reward_avg, 'r-', linewidth=2, label='DQN')
    
    ax2.set_title('Training Progress - Reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance comparison bar chart
    ax3 = plt.subplot(2, 4, 3)
    methods = ['Q-Learning', 'DQN', 'Random']
    best_distances = [q_results['best_distance'], dqn_results['best_distance'], random_best]
    colors = ['blue', 'red', 'gray']
    
    bars = ax3.bar(methods, best_distances, color=colors, alpha=0.7)
    ax3.set_title('Best Distance Comparison')
    ax3.set_ylabel('Distance')
    
    # Add value labels on bars
    for bar, distance in zip(bars, best_distances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{distance:.2f}', ha='center', va='bottom')
    
    ax3.grid(True, alpha=0.3)
    
    # Training time comparison
    ax4 = plt.subplot(2, 4, 4)
    training_times = [q_results['training_time'], dqn_results['training_time']]
    method_names = ['Q-Learning', 'DQN']
    colors = ['blue', 'red']
    
    bars = ax4.bar(method_names, training_times, color=colors, alpha=0.7)
    ax4.set_title('Training Time Comparison')
    ax4.set_ylabel('Time (seconds)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # Q-Learning best solution
    ax5 = plt.subplot(2, 4, 5)
    q_env = q_results['env']
    q_path = q_results['best_path']
    
    ax5.scatter(q_env.cities[:, 0], q_env.cities[:, 1], c='red', s=100, zorder=3)
    for i, (x, y) in enumerate(q_env.cities):
        ax5.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=10)
    
    if q_path and len(q_path) == q_env.num_cities:
        for i in range(len(q_path) - 1):
            city1, city2 = q_path[i], q_path[i + 1]
            ax5.plot([q_env.cities[city1, 0], q_env.cities[city2, 0]],
                    [q_env.cities[city1, 1], q_env.cities[city2, 1]], 'b-', linewidth=2)
        
        # Return to start
        last_city = q_path[-1]
        ax5.plot([q_env.cities[last_city, 0], q_env.cities[0, 0]],
                [q_env.cities[last_city, 1], q_env.cities[0, 1]], 'b--', linewidth=2)
    
    ax5.set_title(f'Q-Learning Best Solution\nDistance: {q_results["best_distance"]:.2f}')
    ax5.grid(True, alpha=0.3)
    
    # DQN best solution
    ax6 = plt.subplot(2, 4, 6)
    dqn_env = dqn_results['env']
    dqn_path = dqn_results['best_path']
    
    ax6.scatter(dqn_env.cities[:, 0], dqn_env.cities[:, 1], c='red', s=100, zorder=3)
    for i, (x, y) in enumerate(dqn_env.cities):
        ax6.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=10)
    
    if dqn_path and len(dqn_path) == dqn_env.num_cities:
        for i in range(len(dqn_path) - 1):
            city1, city2 = dqn_path[i], dqn_path[i + 1]
            ax6.plot([dqn_env.cities[city1, 0], dqn_env.cities[city2, 0]],
                    [dqn_env.cities[city1, 1], dqn_env.cities[city2, 1]], 'r-', linewidth=2)
        
        # Return to start
        last_city = dqn_path[-1]
        ax6.plot([dqn_env.cities[last_city, 0], dqn_env.cities[0, 0]],
                [dqn_env.cities[last_city, 1], dqn_env.cities[0, 1]], 'r--', linewidth=2)
    
    ax6.set_title(f'DQN Best Solution\nDistance: {dqn_results["best_distance"]:.2f}')
    ax6.grid(True, alpha=0.3)
    
    # Convergence comparison
    ax7 = plt.subplot(2, 4, (7, 8))
    
    # Calculate convergence (distance to best solution over time)
    q_convergence = []
    q_best_so_far = float('inf')
    for dist in q_distances:
        if dist < q_best_so_far and dist > 0:
            q_best_so_far = dist
        q_convergence.append(q_best_so_far)
    
    dqn_convergence = []
    dqn_best_so_far = float('inf')
    for dist in dqn_distances:
        if dist < dqn_best_so_far and dist > 0:
            dqn_best_so_far = dist
        dqn_convergence.append(dqn_best_so_far)
    
    ax7.plot(q_convergence, 'b-', linewidth=2, label='Q-Learning')
    ax7.plot(dqn_convergence, 'r-', linewidth=2, label='DQN')
    ax7.axhline(y=random_avg, color='gray', linestyle='--', label='Random Average')
    
    ax7.set_title('Convergence Comparison')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Best Distance So Far')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run comparison
    q_results, dqn_results, random_avg, random_best = compare_methods(
        num_episodes=3000,
        num_cities=10,
        seed=42
    ) 