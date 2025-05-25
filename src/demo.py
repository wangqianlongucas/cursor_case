#!/usr/bin/env python3
"""
Demo script showing how to use the unified TSP environment with different algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
from unified_env import UnifiedTSPEnvironment

def demo_environment_modes():
    """Demonstrate the difference between simple and enhanced modes."""
    print("🚀 TSP强化学习项目演示")
    print("="*60)
    
    # Create environments
    print("\n1. 创建统一环境")
    env_simple = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='simple')
    env_enhanced = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='enhanced')
    
    print(f"   Simple模式状态维度: {env_simple.get_state_size()}")
    print(f"   Enhanced模式状态维度: {env_enhanced.get_state_size()}")
    
    # Show state representations
    print("\n2. 状态表示对比")
    state_simple = env_simple.reset()
    state_enhanced = env_enhanced.reset()
    
    print(f"   Simple状态: {state_simple}")
    print(f"   Enhanced状态形状: {state_enhanced.shape}")
    print(f"   Enhanced状态类型: {type(state_enhanced)}")
    
    # Show action spaces
    print("\n3. 动作空间")
    valid_actions = env_simple.get_valid_actions()
    action_mask = env_enhanced.get_action_mask()
    
    print(f"   有效动作: {valid_actions}")
    print(f"   动作掩码: {action_mask}")
    
    return env_simple, env_enhanced

def demo_random_agent(env, num_episodes=5):
    """Demonstrate a random agent playing TSP."""
    print(f"\n4. 随机智能体演示 ({env.state_mode}模式)")
    print("-" * 40)
    
    distances = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_distance = 0
        done = False
        steps = 0
        
        print(f"   Episode {episode + 1}: ", end="")
        
        while not done and steps < env.num_cities:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action = np.random.choice(valid_actions)
            state, reward, done, info = env.step(action)
            
            if info["valid"]:
                steps += 1
                print(f"{action}→", end="")
            else:
                break
        
        if len(env.visited_cities) == env.num_cities:
            distances.append(env.total_distance)
            print(f"0 (距离: {env.total_distance:.1f})")
        else:
            print("未完成")
    
    if distances:
        print(f"   平均距离: {np.mean(distances):.1f}")
        print(f"   最佳距离: {np.min(distances):.1f}")
    
    return distances

def demo_visualization(env):
    """Demonstrate environment visualization."""
    print(f"\n5. 可视化演示")
    print("-" * 40)
    
    # Reset and take a few random steps
    env.reset()
    
    # Take some random actions to create a partial path
    for _ in range(3):
        valid_actions = env.get_valid_actions()
        if valid_actions:
            action = np.random.choice(valid_actions)
            env.step(action)
    
    print(f"   当前路径: {env.visited_cities}")
    print(f"   当前距离: {env.total_distance:.1f}")
    print(f"   剩余城市: {env.get_valid_actions()}")
    
    # Show the visualization
    env.render(title_suffix="Demo")

def demo_algorithm_compatibility():
    """Demonstrate how different algorithms would use the environment."""
    print(f"\n6. 算法兼容性演示")
    print("-" * 40)
    
    # Q-learning style usage
    print("   Q-learning使用方式:")
    env_q = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='simple')
    state = env_q.reset()
    print(f"     状态: {state}")
    print(f"     状态键: {(tuple(state), env_q.current_city)}")  # Q-table key
    
    # DQN style usage  
    print("\n   DQN使用方式:")
    env_dqn = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='enhanced')
    state = env_dqn.reset()
    print(f"     状态形状: {state.shape}")
    print(f"     状态范围: [{state.min():.2f}, {state.max():.2f}]")
    
    # PPO style usage
    print("\n   PPO使用方式:")
    env_ppo = UnifiedTSPEnvironment(num_cities=5, seed=42, state_mode='enhanced')
    state = env_ppo.reset()
    action_mask = env_ppo.get_action_mask()
    print(f"     状态形状: {state.shape}")
    print(f"     动作掩码: {action_mask}")
    print(f"     掩码动作数: {action_mask.sum()}")

def demo_performance_comparison():
    """Demonstrate performance comparison setup."""
    print(f"\n7. 性能对比设置")
    print("-" * 40)
    
    # Show how to set up fair comparison
    seed = 42
    num_cities = 8
    
    print(f"   问题设置: {num_cities}城市, 种子={seed}")
    
    # Create environments for each algorithm
    env_q = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='simple')
    env_dqn = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='enhanced')
    env_ppo = UnifiedTSPEnvironment(num_cities=num_cities, seed=seed, state_mode='enhanced')
    
    # Verify they have the same problem instance
    cities_match_dqn = np.allclose(env_q.cities, env_dqn.cities)
    cities_match_ppo = np.allclose(env_q.cities, env_ppo.cities)
    
    print(f"   城市位置一致性: Q-learning vs DQN: {cities_match_dqn}")
    print(f"   城市位置一致性: Q-learning vs PPO: {cities_match_ppo}")
    
    # Show state size differences
    print(f"   状态维度对比:")
    print(f"     Q-learning: {env_q.get_state_size()}")
    print(f"     DQN: {env_dqn.get_state_size()}")
    print(f"     PPO: {env_ppo.get_state_size()}")

def main():
    """Run the complete demo."""
    # Demo 1: Environment modes
    env_simple, env_enhanced = demo_environment_modes()
    
    # Demo 2: Random agents
    print("\n" + "="*60)
    distances_simple = demo_random_agent(env_simple, num_episodes=3)
    distances_enhanced = demo_random_agent(env_enhanced, num_episodes=3)
    
    # Demo 3: Visualization
    print("\n" + "="*60)
    demo_visualization(env_enhanced)
    
    # Demo 4: Algorithm compatibility
    print("\n" + "="*60)
    demo_algorithm_compatibility()
    
    # Demo 5: Performance comparison setup
    print("\n" + "="*60)
    demo_performance_comparison()
    
    # Summary
    print("\n" + "="*60)
    print("🎉 演示完成!")
    print("\n📚 接下来可以尝试:")
    print("   • python src/train.py           - Q-learning训练")
    print("   • python src/dqn_train.py       - DQN训练")
    print("   • python src/ppo_train.py       - PPO训练")
    print("   • python src/compare_all_methods.py - 全面对比")
    print("\n💡 提示: 查看SETUP_GUIDE.md了解详细配置说明")

if __name__ == "__main__":
    main() 