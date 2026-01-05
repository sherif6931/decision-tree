class Center:
    def __init__(self, q=0, r=0):
        self.q, self.r = q, r
    
    def __eq__(self, other):
        if not isinstance(other, Center):
            return NotImplemented
        
        return (self.q, self.r) == (other.q, other.r)
    
    def __iter__(self):
        return iter((self.q,self.r))
    
    def __repr__(self):
        return f"{{q={self.q},r={self.r}}}."