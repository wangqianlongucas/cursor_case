import numpy as np
import torch

class DQNTSPEnvironment:
    def __init__(self, num_cities=10, seed=None):
        """
        Enhanced TSP environment for DQN with richer state representation
        
        Args:
            num_cities (int): Number of cities in the TSP
            seed (int): Random seed for reproducibility
        """
        self.num_cities = num_cities
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random city coordinates
        self.cities = np.random.rand(num_cities, 2) * 100
        self.distance_matrix = self._calculate_distance_matrix()
        self.max_distance = np.max(self.distance_matrix)
        self.min_distance = np.min(self.distance_matrix[self.distance_matrix > 0])
        
        # Precompute features for efficiency
        self._precompute_features()
        self.reset()
    
    def _calculate_distance_matrix(self):
        """Calculate the distance matrix between all cities."""
        distance_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                distance_matrix[i][j] = np.sqrt(
                    np.sum((self.cities[i] - self.cities[j]) ** 2)
                )
        return distance_matrix
    
    def _precompute_features(self):
        """Precompute features for enhanced state representation."""
        # Nearest neighbor distances for each city
        self.nearest_neighbors = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            sorted_indices = np.argsort(self.distance_matrix[i])
            self.nearest_neighbors[i] = sorted_indices
        
        # City coordinates normalized
        self.normalized_cities = (self.cities - np.mean(self.cities, axis=0)) / np.std(self.cities, axis=0)
        
        # Distance statistics for each city
        self.city_distance_stats = np.zeros((self.num_cities, 3))  # min, mean, max distances
        for i in range(self.num_cities):
            distances = self.distance_matrix[i][self.distance_matrix[i] > 0]
            self.city_distance_stats[i] = [np.min(distances), np.mean(distances), np.max(distances)]
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_city = 0
        self.visited_cities = [self.current_city]
        self.total_distance = 0
        self.step_count = 0
        return self._get_enhanced_state()
    
    def _get_enhanced_state(self):
        """
        Get enhanced state representation for DQN.
        
        Returns:
            torch.Tensor: Enhanced state vector
        """
        state_features = []
        
        # 1. Basic visited cities (one-hot encoding)
        visited_mask = np.zeros(self.num_cities)
        for city in self.visited_cities:
            visited_mask[city] = 1
        state_features.extend(visited_mask)
        
        # 2. Current city one-hot encoding
        current_city_onehot = np.zeros(self.num_cities)
        current_city_onehot[self.current_city] = 1
        state_features.extend(current_city_onehot)
        
        # 3. Distances from current city to all unvisited cities (normalized)
        distances_to_unvisited = np.zeros(self.num_cities)
        for i in range(self.num_cities):
            if i not in self.visited_cities:
                distances_to_unvisited[i] = self.distance_matrix[self.current_city][i] / self.max_distance
        state_features.extend(distances_to_unvisited)
        
        # 4. Relative positions of unvisited cities
        current_pos = self.normalized_cities[self.current_city]
        relative_positions = np.zeros(self.num_cities * 2)
        for i in range(self.num_cities):
            if i not in self.visited_cities:
                rel_pos = self.normalized_cities[i] - current_pos
                relative_positions[i*2:(i+1)*2] = rel_pos
        state_features.extend(relative_positions)
        
        # 5. Tour progress features
        progress_features = [
            len(self.visited_cities) / self.num_cities,  # completion ratio
            self.step_count / self.num_cities,  # step ratio
            self.total_distance / (self.max_distance * self.num_cities) if self.total_distance > 0 else 0,  # distance ratio
        ]
        state_features.extend(progress_features)
        
        # 6. Nearest neighbor information
        nn_features = np.zeros(self.num_cities)
        for i in range(min(3, self.num_cities)):  # Top 3 nearest neighbors
            if i < len(self.nearest_neighbors[self.current_city]):
                nn_city = int(self.nearest_neighbors[self.current_city][i])
                if nn_city not in self.visited_cities:
                    nn_features[nn_city] = (3 - i) / 3  # Weight by proximity rank
        state_features.extend(nn_features)
        
        # 7. Statistical features
        remaining_cities = [i for i in range(self.num_cities) if i not in self.visited_cities]
        if remaining_cities:
            remaining_distances = [self.distance_matrix[self.current_city][i] for i in remaining_cities]
            stat_features = [
                np.min(remaining_distances) / self.max_distance,
                np.mean(remaining_distances) / self.max_distance,
                np.max(remaining_distances) / self.max_distance,
                len(remaining_cities) / self.num_cities
            ]
        else:
            stat_features = [0, 0, 0, 0]
        state_features.extend(stat_features)
        
        return torch.FloatTensor(state_features)
    
    def get_state_size(self):
        """Get the size of the state vector."""
        dummy_state = self._get_enhanced_state()
        return len(dummy_state)
    
    def step(self, action):
        """
        Take a step in the environment with enhanced reward shaping
        
        Args:
            action (int): The city to visit next
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action in self.visited_cities:
            return self._get_enhanced_state(), -50, True, {"valid": False}
        
        # Calculate distance and update state
        distance = self.distance_matrix[self.current_city][action]
        self.total_distance += distance
        self.step_count += 1
        
        prev_city = self.current_city
        self.current_city = action
        self.visited_cities.append(action)
        
        # Check if tour is complete
        done = len(self.visited_cities) == self.num_cities
        
        # Enhanced reward shaping
        reward = self._calculate_reward(distance, done, prev_city, action)
        
        return self._get_enhanced_state(), reward, done, {"valid": True}
    
    def _calculate_reward(self, distance, done, prev_city, current_city):
        """Calculate enhanced reward with multiple components."""
        reward = 0
        
        # 1. Distance penalty (normalized)
        distance_penalty = -(distance / self.max_distance) * 10
        reward += distance_penalty
        
        # 2. Completion bonus
        if done:
            return_distance = self.distance_matrix[current_city][0]
            self.total_distance += return_distance
            completion_bonus = 50 - (self.total_distance / (self.max_distance * self.num_cities)) * 20
            reward += completion_bonus
        
        # 3. Progress reward (small positive reward for making progress)
        progress_reward = 0.5
        reward += progress_reward
        
        # 4. Nearest neighbor bonus (encourage visiting nearby cities)
        remaining_cities = [i for i in range(self.num_cities) if i not in self.visited_cities]
        if remaining_cities:
            min_remaining_distance = min(self.distance_matrix[current_city][i] for i in remaining_cities)
            if min_remaining_distance < distance:
                nn_bonus = 1.0
                reward += nn_bonus
        
        # 5. Backtracking penalty (discourage returning to recently visited areas)
        if len(self.visited_cities) > 2:
            recent_cities = self.visited_cities[-3:-1]
            for recent_city in recent_cities:
                if self.distance_matrix[current_city][recent_city] < self.min_distance * 2:
                    backtrack_penalty = -2.0
                    reward += backtrack_penalty
                    break
        
        return reward
    
    def get_valid_actions(self):
        """Get list of valid actions (unvisited cities)."""
        return [i for i in range(self.num_cities) if i not in self.visited_cities]
    
    def render(self):
        """Render the current state of the environment."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=150, zorder=3)
        
        # Add city labels
        for i, (x, y) in enumerate(self.cities):
            color = 'blue' if i in self.visited_cities else 'black'
            weight = 'bold' if i == self.current_city else 'normal'
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=12, color=color, weight=weight)
        
        # Plot the path
        for i in range(len(self.visited_cities) - 1):
            city1 = self.visited_cities[i]
            city2 = self.visited_cities[i + 1]
            plt.plot([self.cities[city1, 0], self.cities[city2, 0]],
                    [self.cities[city1, 1], self.cities[city2, 1]], 'b-', linewidth=3)
            
            # Add direction arrows
            mid_x = (self.cities[city1, 0] + self.cities[city2, 0]) / 2
            mid_y = (self.cities[city1, 1] + self.cities[city2, 1]) / 2
            dx = self.cities[city2, 0] - self.cities[city1, 0]
            dy = self.cities[city2, 1] - self.cities[city1, 1]
            plt.arrow(mid_x, mid_y, dx*0.1, dy*0.1, head_width=2, head_length=2, 
                     fc='blue', ec='blue', alpha=0.7)
        
        # Plot return to start if tour is complete
        if len(self.visited_cities) == self.num_cities:
            last_city = self.visited_cities[-1]
            plt.plot([self.cities[last_city, 0], self.cities[0, 0]],
                    [self.cities[last_city, 1], self.cities[0, 1]], 'b--', linewidth=3)
        
        plt.title(f'TSP Tour (Distance: {self.total_distance:.2f}, Cities: {len(self.visited_cities)}/{self.num_cities})')
        plt.grid(True, alpha=0.3)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show() 