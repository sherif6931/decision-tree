import numpy as np
from entity import Entity, Map, Hex, Center, CONST_direction

class Survivor(Entity):
    def __init__(self, map):
        super().__init__(map)
        self.hexEntity = map.hexes[Hex(-12,6,6)]
        self.position = self.entity_position()
        self.sane = True
        self.sanity = 1.0

    def eat(self, vl):
        self.stats += ([0, max(self.hunger - vl, 0), min(self.stamina + ((1/3) * vl), 100)])
        
    def decide(self):
        if not self.sane:
            return super().move(CONST_direction[np.random.randint(0,6)])
        else:
            return None
    
    def decay_sanity(self, amount=0.01):
        self.sanity = max(0.2, self.sanity - amount)
        if self.sanity < 0.5:
            self.sane = False
    
    def __repr__(self):
        return f"Entity={type(self).__name__}. Health={self.points}. Sanity={self.sanity}."