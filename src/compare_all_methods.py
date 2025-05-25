import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd

# Import unified environment
from unified_env import UnifiedTSPEnvironment

# Import Q-learning components
from q_learning import QLearningAgent

# Import DQN components  
from dqn_agent import DQNAgent

# Import PPO components
from ppo_agent import PPOAgent

def train_q_learning(num_episodes, num_cities, seed):
    """Train Q-learning agent using unified environment."""
    env = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='simple')
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
        'env': env,
        'agent': agent
    }

def train_dqn(num_episodes, num_cities, seed):
    """Train DQN agent using unified environment."""
    env = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='enhanced')
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
        'env': env,
        'agent': agent
    }

def train_ppo(num_episodes, num_cities, seed, update_frequency=1024):
    """Train PPO agent using unified environment."""
    env = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='enhanced')
    state_size = env.get_state_size()
    agent = PPOAgent(
        state_size=state_size,
        action_size=num_cities,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        buffer_size=update_frequency
    )
    
    episode_rewards = []
    episode_distances = []
    best_distance = float('inf')
    best_path = None
    
    start_time = time.time()
    step_count = 0
    
    for episode in tqdm(range(num_episodes), desc="Training PPO"):
        state = env.reset()
        total_reward = 0
        done = False
        episode_length = 0
        
        while not done and episode_length < num_cities * 2:
            action_mask = env.get_action_mask()
            action = agent.get_action(state, action_mask)
            
            next_state, reward, done, info = env.step(action)
            
            if info["valid"]:
                agent.store_transition(state, action, reward, done, action_mask)
                total_reward += reward
                state = next_state
                step_count += 1
                episode_length += 1
                
                # Update policy if buffer is full
                if step_count % update_frequency == 0:
                    next_value = 0
                    if not done:
                        next_action_mask = env.get_action_mask()
                        with torch.no_grad():
                            _, _, next_value = agent.actor_critic.get_action_and_value(
                                torch.FloatTensor(next_state).unsqueeze(0).to(agent.device),
                                torch.FloatTensor(next_action_mask).unsqueeze(0).to(agent.device)
                            )
                        next_value = next_value.cpu().item()
                    
                    agent.update(next_value)
            else:
                break
        
        episode_rewards.append(total_reward)
        episode_distances.append(env.total_distance)
        
        if env.total_distance < best_distance and len(env.visited_cities) == num_cities:
            best_distance = env.total_distance
            best_path = env.visited_cities.copy()
    
    training_time = time.time() - start_time
    
    return {
        'method': 'PPO',
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'best_distance': best_distance,
        'best_path': best_path,
        'training_time': training_time,
        'env': env,
        'agent': agent
    }

def test_agent(agent, env, method, num_tests=100):
    """Test a trained agent."""
    test_distances = []
    test_rewards = []
    successful_tours = 0
    
    for _ in range(num_tests):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.num_cities * 2:
            if method == 'Q-Learning':
                valid_actions = env.get_valid_actions()
                action = agent.get_action(state, env.current_city, valid_actions, training=False)
            elif method == 'DQN':
                valid_actions = env.get_valid_actions()
                action = agent.get_action(state, valid_actions, training=False)
            elif method == 'PPO':
                action_mask = env.get_action_mask()
                action = agent.get_action(state, action_mask)
            
            if action is None:
                break
                
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
    
    return {
        'distances': test_distances,
        'rewards': test_rewards,
        'success_rate': successful_tours / num_tests,
        'avg_distance': np.mean(test_distances) if test_distances else float('inf'),
        'best_distance': np.min(test_distances) if test_distances else float('inf')
    }

def calculate_random_baseline(num_cities, seed, num_tests=1000):
    """Calculate random baseline performance."""
    env = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='simple')
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
        
        if len(env.visited_cities) == num_cities:
            random_distances.append(env.total_distance)
    
    return {
        'avg_distance': np.mean(random_distances) if random_distances else float('inf'),
        'best_distance': np.min(random_distances) if random_distances else float('inf'),
        'distances': random_distances
    }

def compare_all_methods(num_episodes=5000, num_cities=10, seed=42, num_tests=100):
    """Compare all three methods on TSP."""
    print(f"Comprehensive TSP Comparison")
    print(f"Cities: {num_cities}, Episodes: {num_episodes}, Seed: {seed}")
    print("="*80)
    
    # Train all methods
    print("\n1. Training Q-Learning...")
    q_results = train_q_learning(num_episodes, num_cities, seed)
    
    print("\n2. Training DQN...")
    dqn_results = train_dqn(num_episodes, num_cities, seed)
    
    print("\n3. Training PPO...")
    ppo_results = train_ppo(num_episodes, num_cities, seed)
    
    # Calculate random baseline
    print("\n4. Calculating random baseline...")
    random_baseline = calculate_random_baseline(num_cities, seed)
    
    # Test all methods
    print("\n5. Testing trained agents...")
    q_test = test_agent(q_results['agent'], q_results['env'], 'Q-Learning', num_tests)
    dqn_test = test_agent(dqn_results['agent'], dqn_results['env'], 'DQN', num_tests)
    ppo_test = test_agent(ppo_results['agent'], ppo_results['env'], 'PPO', num_tests)
    
    # Compile results
    results = {
        'Q-Learning': {**q_results, 'test_results': q_test},
        'DQN': {**dqn_results, 'test_results': dqn_test},
        'PPO': {**ppo_results, 'test_results': ppo_test},
        'Random': random_baseline
    }
    
    # Print comparison table
    print_comparison_table(results)
    
    # Plot comprehensive comparison
    plot_comprehensive_comparison(results)
    
    return results

def print_comparison_table(results):
    """Print a comprehensive comparison table."""
    print("\n" + "="*100)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("="*100)
    
    # Create comparison data
    methods = ['Q-Learning', 'DQN', 'PPO', 'Random']
    data = []
    
    for method in methods:
        if method == 'Random':
            row = [
                method,
                '-',
                f"{results[method]['avg_distance']:.2f}",
                f"{results[method]['best_distance']:.2f}",
                '-',
                '-',
                '-'
            ]
        else:
            test_results = results[method]['test_results']
            row = [
                method,
                f"{results[method]['training_time']:.1f}s",
                f"{test_results['avg_distance']:.2f}",
                f"{test_results['best_distance']:.2f}",
                f"{results[method]['best_distance']:.2f}",
                f"{test_results['success_rate']*100:.1f}%",
                f"{results[method]['best_path']}"
            ]
        data.append(row)
    
    # Create DataFrame for nice formatting
    df = pd.DataFrame(data, columns=[
        'Method', 'Training Time', 'Test Avg Distance', 'Test Best Distance', 
        'Training Best', 'Success Rate', 'Best Path'
    ])
    
    print(df.to_string(index=False))
    
    # Calculate improvements
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    random_avg = results['Random']['avg_distance']
    
    for method in ['Q-Learning', 'DQN', 'PPO']:
        test_avg = results[method]['test_results']['avg_distance']
        if test_avg > 0:
            improvement = random_avg / test_avg
            print(f"{method:12} vs Random: {improvement:.2f}x improvement")
    
    # Best method analysis
    best_method = min(['Q-Learning', 'DQN', 'PPO'], 
                     key=lambda m: results[m]['test_results']['avg_distance'])
    print(f"\nBest performing method: {best_method}")
    print(f"Best distance achieved: {results[best_method]['test_results']['best_distance']:.2f}")

def plot_comprehensive_comparison(results):
    """Plot comprehensive comparison of all methods."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    methods = ['Q-Learning', 'DQN', 'PPO']
    colors = ['blue', 'red', 'green']
    
    # 1. Training curves - Distance
    for i, method in enumerate(methods):
        episodes = range(len(results[method]['episode_distances']))
        distances = results[method]['episode_distances']
        axes[0, 0].plot(episodes, smooth_curve(distances, 100), 
                       color=colors[i], label=method, linewidth=2)
    
    axes[0, 0].axhline(y=results['Random']['avg_distance'], color='black', 
                      linestyle='--', label='Random Baseline')
    axes[0, 0].set_title('Training Progress - Distance')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Distance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Training curves - Reward
    for i, method in enumerate(methods):
        episodes = range(len(results[method]['episode_rewards']))
        rewards = results[method]['episode_rewards']
        axes[0, 1].plot(episodes, smooth_curve(rewards, 100), 
                       color=colors[i], label=method, linewidth=2)
    
    axes[0, 1].set_title('Training Progress - Reward')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Performance comparison
    method_names = methods + ['Random']
    avg_distances = [results[m]['test_results']['avg_distance'] if m != 'Random' 
                    else results[m]['avg_distance'] for m in method_names]
    best_distances = [results[m]['test_results']['best_distance'] if m != 'Random' 
                     else results[m]['best_distance'] for m in method_names]
    
    x = np.arange(len(method_names))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, avg_distances, width, label='Average', alpha=0.8)
    axes[0, 2].bar(x + width/2, best_distances, width, label='Best', alpha=0.8)
    axes[0, 2].set_title('Performance Comparison')
    axes[0, 2].set_xlabel('Method')
    axes[0, 2].set_ylabel('Distance')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(method_names)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Training time comparison
    training_times = [results[m]['training_time'] for m in methods]
    axes[1, 0].bar(methods, training_times, color=colors, alpha=0.8)
    axes[1, 0].set_title('Training Time Comparison')
    axes[1, 0].set_xlabel('Method')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Success rate comparison
    success_rates = [results[m]['test_results']['success_rate'] * 100 for m in methods]
    axes[1, 1].bar(methods, success_rates, color=colors, alpha=0.8)
    axes[1, 1].set_title('Success Rate Comparison')
    axes[1, 1].set_xlabel('Method')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Best solution visualization
    best_method = min(methods, key=lambda m: results[m]['test_results']['best_distance'])
    env = results[best_method]['env']
    env.visited_cities = results[best_method]['best_path']
    env.current_city = results[best_method]['best_path'][-1]
    env.total_distance = results[best_method]['best_distance']
    
    axes[1, 2].scatter(env.cities[:, 0], env.cities[:, 1], c='red', s=150, zorder=3)
    
    # Add city labels
    for i, (x, y) in enumerate(env.cities):
        axes[1, 2].annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=10, color='blue', weight='bold')
    
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
    
    axes[1, 2].set_title(f'Best Solution ({best_method})\nDistance: {env.total_distance:.2f}')
    axes[1, 2].set_xlabel('X Coordinate')
    axes[1, 2].set_ylabel('Y Coordinate')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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

if __name__ == "__main__":
    # Configuration
    config = {
        'num_episodes': 5000,
        'num_cities': 10,
        'seed': 42,
        'num_tests': 100
    }
    
    print("Starting Comprehensive TSP Method Comparison...")
    print(f"Configuration: {config}")
    
    # Run comparison
    results = compare_all_methods(**config)
    
    print("\nComparison completed!")
    print("Check the plots and tables above for detailed analysis.") 