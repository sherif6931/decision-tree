from entity import Entity, Map, Hex, CONST_direction, Center

class Rescuer(Entity):
    def __init__(self, map):
        super().__init__(map)
    
    def seek(self):
        return None # Path-finder later.

    def __repr__(self):
        return f"Entity={type(self).__name__}. Health={self.points}."