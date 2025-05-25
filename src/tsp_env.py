import numpy as np

class TSPEnvironment:
    def __init__(self, num_cities=10, seed=None):
        """
        Initialize the TSP environment
        
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
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_city = 0
        self.visited_cities = [self.current_city]
        self.total_distance = 0
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """Get the current state representation."""
        state = np.zeros(self.num_cities)
        for city in self.visited_cities:
            state[city] = 1
        return state
    
    def step(self, action):
        """
        Take a step in the environment with improved reward shaping
        
        Args:
            action (int): The city to visit next
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action in self.visited_cities:
            return self._get_state(), -100, True, {"valid": False}
        
        # Add the distance from current city to next city
        distance = self.distance_matrix[self.current_city][action]
        self.total_distance += distance
        self.step_count += 1
        
        # Update current city and visited cities
        prev_city = self.current_city
        self.current_city = action
        self.visited_cities.append(action)
        
        # Check if all cities have been visited
        done = len(self.visited_cities) == self.num_cities
        
        # Improved reward shaping
        if done:
            # Add return distance to starting city
            return_distance = self.distance_matrix[action][0]
            self.total_distance += return_distance
            # Bonus for completing the tour, penalty for total distance
            reward = 100 - (self.total_distance / self.max_distance) * 10
        else:
            # Negative reward proportional to distance, with small step penalty
            reward = -(distance / self.max_distance) * 10 - 0.1
        
        return self._get_state(), reward, done, {"valid": True}
    
    def get_valid_actions(self):
        """Get list of valid actions (unvisited cities)."""
        return [i for i in range(self.num_cities) if i not in self.visited_cities]
    
    def render(self):
        """Render the current state of the environment."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        
        # Add city labels
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot the path
        for i in range(len(self.visited_cities) - 1):
            city1 = self.visited_cities[i]
            city2 = self.visited_cities[i + 1]
            plt.plot([self.cities[city1, 0], self.cities[city2, 0]],
                    [self.cities[city1, 1], self.cities[city2, 1]], 'b-', linewidth=2)
        
        # Plot return to start if tour is complete
        if len(self.visited_cities) == self.num_cities:
            last_city = self.visited_cities[-1]
            plt.plot([self.cities[last_city, 0], self.cities[0, 0]],
                    [self.cities[last_city, 1], self.cities[0, 1]], 'b--', linewidth=2)
        
        plt.title(f'Total Distance: {self.total_distance:.2f}')
        plt.grid(True, alpha=0.3)
        plt.show() 