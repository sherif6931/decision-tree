import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from core.map import Map, Hex, Center
from utils.astar import astar, rationalBias

class SurvivorDecisionTree:
    def __init__(self, map: Map):
        self.map = map
        self.model = DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42)
        self.trained = False
    
    def get_features(self, survivor_hex, rescuer_hex):
        features = []
        dist = self.map.hex_distance(survivor_hex, rescuer_hex)
        features.append(dist)
        
        neighbors = self.map.neighbor_hex(survivor_hex)
        
        for nb in neighbors:
            if nb is None:
                features.append(10.0)
                features.append(dist)
            else:
                cost = self.map.costs[nb]
                if cost == float('-inf'):
                    features.append(10.0)
                    features.append(dist)
                else:
                    features.append(float(cost))
                    features.append(self.map.hex_distance(nb, rescuer_hex))
        
        return np.array(features)
    
    def generate_training_data(self, n_samples=2000):
        X_train = []
        y_train = []
        
        valid_hexes = [h for h in self.map.hexes.values() if self.map.costs[h] != float('-inf')]
        
        for _ in range(n_samples):
            survivor_hex = np.random.choice(valid_hexes)
            rescuer_candidates = [h for h in valid_hexes if self.map.hex_distance(h, survivor_hex) > 5]
            
            if not rescuer_candidates:
                continue
                
            rescuer_hex = np.random.choice(rescuer_candidates)
            features = self.get_features(survivor_hex, rescuer_hex)
            action = rationalBias(survivor_hex, self.map, degree_of_bias=7)
            
            X_train.append(features)
            y_train.append(action)
        
        return np.array(X_train), np.array(y_train)
    
    def train(self, n_samples=2000):
        X, y = self.generate_training_data(n_samples)
        self.model.fit(X, y)
        self.trained = True
        print(f"Training complete. Score: {self.model.score(X, y):.3f}")
    
    def decide(self, survivor_hex, rescuer_hex, sanity=1.0):
        if not self.trained:
            self.train()
        
        if sanity < 0.5 and np.random.random() > sanity:
            neighbors = self.map.neighbor_hex(survivor_hex)
            valid = [i for i, nb in enumerate(neighbors) if nb is not None and self.map.costs[nb] != float('-inf')]
            return np.random.choice(valid) if valid else None
        
        features = self.get_features(survivor_hex, rescuer_hex)
        action = self.model.predict(features.reshape(1, -1))[0]
        
        neighbors = self.map.neighbor_hex(survivor_hex)
        if action < 6 and neighbors[action] is not None:
            if self.map.costs[neighbors[action]] != float('-inf'):
                return int(action)
        
        return rationalBias(survivor_hex, self.map, degree_of_bias=7)

class SimpleBelief:
    def __init__(self, map: Map):
        self.map = map
        self.belief = {}
        
        valid = [h for h in map.hexes.values() if map.costs[h] != float('-inf')]
        prob = 1.0 / len(valid)
        for h in valid:
            self.belief[h] = prob
    
    def observe(self, rescuer_hex, visible_hexes, survivor_seen=False, survivor_hex=None):
        if survivor_seen and survivor_hex:
            for h in self.belief:
                self.belief[h] = 0.0
            self.belief[survivor_hex] = 1.0
        else:
            for h in visible_hexes:
                if h in self.belief:
                    self.belief[h] = 0.0
            
            total = sum(self.belief.values())
            if total > 0:
                for h in self.belief:
                    self.belief[h] /= total
    
    def predict_movement(self):
        new_belief = {}
        
        for h in self.belief:
            if self.belief[h] < 1e-6:
                continue
            
            neighbors = self.map.neighbor_hex(h)
            valid_neighbors = [nb for nb in neighbors if nb is not None and self.map.costs[nb] != float('-inf')]
            
            if not valid_neighbors:
                continue
            
            costs = [self.map.costs[nb] for nb in valid_neighbors]
            max_cost = max(costs)
            weights = [max_cost - c + 1 for c in costs]
            total_weight = sum(weights)
            
            for nb, weight in zip(valid_neighbors, weights):
                if nb not in new_belief:
                    new_belief[nb] = 0.0
                new_belief[nb] += self.belief[h] * (weight / total_weight) * 0.7
            
            if h not in new_belief:
                new_belief[h] = 0.0
            new_belief[h] += self.belief[h] * 0.3
        
        self.belief = new_belief
        
        total = sum(self.belief.values())
        if total > 0:
            for h in self.belief:
                self.belief[h] /= total
    
    def get_clusters(self, n_clusters=3):
        significant = [(h, p) for h, p in self.belief.items() if p > 0.001]
        
        if len(significant) < n_clusters:
            return [(h, p) for h, p in sorted(significant, key=lambda x: -x[1])]
        
        coords = np.array([[h.x, h.z] for h, _ in significant])
        probs = np.array([p for _, p in significant])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords, sample_weight=probs)
        
        clusters = []
        for i in range(n_clusters):
            mask = labels == i
            cluster_probs = probs[mask]
            weight = np.sum(cluster_probs)
            
            center_coord = kmeans.cluster_centers_[i]
            center_hex = self.map.hex_round(Hex(int(center_coord[0]), 0, int(center_coord[1])))
            center_hex = Hex(center_hex.x, center_hex.y, center_hex.z)
            
            if center_hex in self.map.hexes:
                clusters.append((self.map.hexes[center_hex], weight))
        
        return clusters

class RescuerAI:
    def __init__(self, map: Map, belief: SimpleBelief):
        self.map = map
        self.belief = belief
    
    def get_best_move(self, rescuer_hex, visited):
        clusters = self.belief.get_clusters(n_clusters=3)
        
        if not clusters:
            return None
        
        neighbors = self.map.neighbor_hex(rescuer_hex)
        valid_moves = [(i, nb) for i, nb in enumerate(neighbors) if nb is not None and self.map.costs[nb] != float('-inf')]
        
        if not valid_moves:
            return None
        
        best_action = None
        best_score = float('-inf')
        
        for action, move_hex in valid_moves:
            score = 0.0
            
            for cluster_hex, weight in clusters:
                dist_before = self.map.hex_distance(rescuer_hex, cluster_hex)
                dist_after = self.map.hex_distance(move_hex, cluster_hex)
                improvement = (dist_before - dist_after) * weight
                score += improvement * 10
            
            if move_hex not in visited:
                score += 2
            
            score -= self.map.costs[move_hex]
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def get_move_with_astar(self, rescuer_hex, visible):
        clusters = self.belief.get_clusters(n_clusters=3)
        
        if not clusters:
            return None
        
        best_first_move = None
        best_score = float('-inf')
        
        for cluster_hex, weight in clusters:
            def heuristic(h1, h2):
                return self.map.hex_distance(h1, h2)
            
            path = astar(rescuer_hex, cluster_hex, self.map, heuristic, visible)
            
            if path and len(path) > 1:
                score = weight * (1.0 / len(path))
                
                if score > best_score:
                    best_score = score
                    first_move_hex = path[1]
                    neighbors = self.map.neighbor_hex(rescuer_hex)
                    for i, nb in enumerate(neighbors):
                        if nb == first_move_hex:
                            best_first_move = i
                            break
        
        return best_first_move


class GameAI:
    def __init__(self, map: Map):
        self.map = map
        self.survivor_ai = SurvivorDecisionTree(map)
        self.belief = SimpleBelief(map)
        self.rescuer_ai = RescuerAI(map, self.belief)
        self.turn_count = 0
    
    def survivor_turn(self, survivor_entity, rescuer_entity, visited):
        survivor_entity.decay_sanity(0.01)
        
        action = self.survivor_ai.decide(
            survivor_entity.hexEntity,
            rescuer_entity.hexEntity,
            sanity=survivor_entity.sanity
        )
        
        if action is not None:
            directions = ['SS', 'SE', 'SW', 'NN', 'NE', 'NW']
            survivor_entity.move(directions[action])
    
    def rescuer_turn(self, rescuer_entity, survivor_entity, visited, visible):
        self.turn_count += 1
        
        survivor_visible = survivor_entity.hexEntity in visible
        self.belief.observe(
            rescuer_entity.hexEntity,
            visible,
            survivor_seen=survivor_visible,
            survivor_hex=survivor_entity.hexEntity if survivor_visible else None
        )
        
        self.belief.predict_movement()
        
        action = self.rescuer_ai.get_move_with_astar(rescuer_entity.hexEntity, visited)
        
        if action is None:
            action = self.rescuer_ai.get_best_move(rescuer_entity.hexEntity, visited)
        
        if action is not None:
            directions = ['SS', 'SE', 'SW', 'NN', 'NE', 'NW']
            rescuer_entity.move(directions[action])
    
    def get_belief_for_visualization(self):
        return self.belief.belief