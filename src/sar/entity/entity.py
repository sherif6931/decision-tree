from collections import namedtuple
import pygame
import numpy as np
from core.map import Map, Hex, Center
from core.config import ENTITY_COLOR, OFFSET

Directions = namedtuple('Direction', ['SS', 'SE', 'SW', 'NN', 'NE', 'NW'])
CONST_direction = Directions(SS=0, SE=1, SW=2, NN=3, NE=4, NW=5)

class Entity:
    def __init__(self, map: Map, color=ENTITY_COLOR):
        self.map = map
        self.hexEntity = self.map.hexes[Hex()]
        self.color = color
        self.points = 100
        self.position = self.entity_position()
        
    def entity_position(self):
        center = self.map.hex_to_screen(self.hexEntity)

        position = np.array([*center])+OFFSET

        return position
        
    def move(self, move_dir):
        if self.points <= 0:
            return None

        move_to_tiles = self.map.neighbor_hex(self.hexEntity)

        i = getattr(CONST_direction, move_dir, None)

        if move_to_tiles[i] == None:
            return None

        self.hexEntity = move_to_tiles[i]

        self.position = self.entity_position()

        return list(move_to_tiles[i])
    
    def hurt(self, vl):
        self.points -= vl

        return self.points