from collections import deque, namedtuple
import pygame
import numpy as np
from map import Map, Hex, Center
from entities.survivor import Survivor
from entities.rescuer import Rescuer
from utils.field_of_view import *
from utils.ai_core import GameAI
from config import WIDTH, HEIGHT, FPS, DEBUG, SIZE, OFFSET, COST_COLORS

if DEBUG:
    from utils.debugger import Debugger

screen = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("SAR Game")

Entities = namedtuple('Entities', ['survivor','rescuer'])

class Game:
    def __init__(self, radius=SIZE):
        pygame.init()
        self.screen = screen
        self.size = radius
        self.screen.fill((0, 0, 0))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, int(self.size*(2/3)))
        self.running = True
        self.debugger = None
        self.removeFog = False
        self.removeRed = False
        self.events_info = []
        self.map = Map(radius)
        self.visible = {}
        self.entities = Entities(Survivor(self.map), Rescuer(self.map))
        self.visited = deque([self.entities.rescuer.hexEntity])
        
        self.ai = GameAI(self.map)
        self.ai.survivor_ai.train(n_samples=2000)
        self.ai_control_survivor = True
        self.ai_control_rescuer = False
        self.show_belief = False

        if DEBUG:
            self.debugger = Debugger(self, False)

    def __get_cursor(self):
        return np.array(pygame.mouse.get_pos())
    
    def select_hex(self):
        point = self.__get_cursor() - OFFSET
        return self.map.screen_to_hex(point)
    
    def discover_hex(self):
        h = self.entities.rescuer.hexEntity
        
        if h in self.visited:
            return None

        self.visited.append(h)  

    def see_hex(self):
        q = self.entities.rescuer.hexEntity
        self.visible = field_of_view(q, self.map, self.visited, limit=-1)  

    def handle_single_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return None
        
        if event.type == pygame.KEYDOWN:
            mapping = {
                pygame.K_w: "NN",
                pygame.K_e: "NE",
                pygame.K_q: "NW",
                pygame.K_s: "SS",
                pygame.K_d: "SE",
                pygame.K_a: "SW",
            }
            
            dir = mapping.get(event.key)
            if dir:
                moved = self.entities.rescuer.move(dir)
                self.discover_hex()
                self.see_hex()

                if moved == None:
                    dir = "nowhere"
                
                if self.ai_control_survivor:
                    self.ai.survivor_turn(self.entities.survivor, self.entities.rescuer, self.visited)

                if self.debugger:
                    self.debugger.get_entity_feed(dir)
            
            if event.key == pygame.K_1:
                self.ai_control_survivor = not self.ai_control_survivor
                print(f"Survivor AI: {'ON' if self.ai_control_survivor else 'OFF'}")
            
            if event.key == pygame.K_2:
                self.ai_control_rescuer = not self.ai_control_rescuer
                print(f"Rescuer AI: {'ON' if self.ai_control_rescuer else 'OFF'}")
            
            if event.key == pygame.K_3:
                self.show_belief = not self.show_belief
                print(f"Belief overlay: {'ON' if self.show_belief else 'OFF'}")
    
    def update_event_info(self):
        for event in pygame.event.get():

            if self.debugger:
                self.debugger.feed(event)
                self.debugger.get_event()

            eventName = pygame.event.event_name(event.type)
            eventInfo = [eventName, self.clock.get_time()]

            if eventName == "KeyDown":
                eventInfo.append(pygame.key.name(event.key))
            elif eventName == "MouseButtonDown":
                eventInfo.append(event.pos)

            self.events_info.append(eventInfo)
            self.handle_single_event(event)

    def handle_events(self):
        self.update_event_info()

    def fog(self, h, color, border_color):
        if self.debugger:
            if self.debugger.toggleOverlay and self.debugger.removeFog:
                return color, border_color

        if not h in self.visible:
            border_color,color = (0,0,0),(0,0,0)

        if h not in self.visited and any(neighbor == h for neighbor in  self.map.neighbor_hex(self.entities.rescuer.hexEntity)):
            if color == (0,0,0):
                color = (30,30,30)
                border_color = (30,30,30)

        return color, border_color
    
    def color_me_red(self, cost, color):
        if self.debugger and self.debugger.toggleOverlay and self.debugger.removeRed:
            return color
        elif self.debugger and self.debugger.toggleOverlay:
            return COST_COLORS[(cost if cost != float('-inf') else 6)]
        else:
            return color

    def draw_map(self):
        for h in self.map.hexes.values():
                border_color = (50,50,50)
                cost = self.map.costs[h]
                vertices, color = self.map.draw_hex(h)

                if self.debugger:
                    color = self.color_me_red(cost,color)

                color, border_color = self.fog(h,color,border_color)

                pygame.draw.polygon(self.screen, color, vertices)
                pygame.draw.polygon(self.screen, border_color, vertices, 1)
    
    def draw_belief_overlay(self):
        if not self.show_belief:
            return
        
        belief = self.ai.get_belief_for_visualization()
        max_prob = max(belief.values()) if belief else 1
        
        for hex, prob in belief.items():
            if prob > 0.01:
                vertices, _ = self.map.draw_hex(hex)
                intensity = int(200 * (prob / max_prob))
                color = (intensity, 0, 255 - intensity)
                pygame.draw.polygon(self.screen, color, vertices, 0)

    def spawn(self):
        for entity in self.entities:
            pygame.draw.circle(self.screen, entity.color, entity.position, radius=int(self.size*(3/5)))

    def run(self):
        while self.running:
            self.handle_events()
            
            if self.ai_control_rescuer:
                self.ai.rescuer_turn(self.entities.rescuer, self.entities.survivor, self.visited, self.visible)
                self.discover_hex()
                self.see_hex()
            
            self.draw_map()
            self.draw_belief_overlay()
            self.spawn()

            if self.debugger:
                self.debugger.overlay()

            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    game = Game()
    game.run()