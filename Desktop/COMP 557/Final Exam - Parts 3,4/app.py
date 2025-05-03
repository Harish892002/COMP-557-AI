import pygame
import sys
import time
from models import Action
from env import EnvironmentBase
from rl_utils import BaseAgent


class GridWorldApp:
    TILE_SIZE = 50
    WALL_THICKNESS = 6
    MARGIN = 2
    FPS = 10

    def __init__(self, env: EnvironmentBase, agent=None):
        pygame.init()
        self.env = env
        self.agent = agent  # type: BaseAgent

        self.screen = pygame.display.set_mode(
            (self.env.width * self.TILE_SIZE,
             self.env.height * self.TILE_SIZE))

        pygame.display.set_caption("Grid World")
        self.clock = pygame.time.Clock()
        self.running = True

        self.agent_img = pygame.image.load('images/robot.png').convert_alpha()
        self.agent_img = pygame.transform.scale(
            self.agent_img, (self.TILE_SIZE - 2 * self.MARGIN,
                             self.TILE_SIZE - 2 * self.MARGIN))
        self.obj_img = pygame.image.load(
            'images/medical-kit.png').convert_alpha()
        self.obj_img = pygame.transform.scale(
            self.obj_img,
            (int(self.TILE_SIZE * 0.7), int(self.TILE_SIZE * 0.7)))

        self.obj_fence_h = pygame.image.load(
            'images/fence.png').convert_alpha()
        self.obj_fence_h = pygame.transform.scale(
            self.obj_fence_h, (self.TILE_SIZE, self.WALL_THICKNESS))
        self.obj_fence_v = pygame.transform.rotate(self.obj_fence_h, 90)

        self.env.reset()

    def draw_grid(self):
        self.screen.fill((255, 255, 255))

        cur_state = self.env.get_current_state()
        cur_loc = cur_state.get_agent_location()
        objs = self.env.get_current_obj_locations()

        for loc in objs:
            obj_w, obj_h = self.obj_img.get_size()
            scr_x = loc[1] * self.TILE_SIZE + self.MARGIN
            scr_y = (loc[0] + 1) * self.TILE_SIZE - obj_h - self.MARGIN
            obj_screen_pos = (scr_x, scr_y)
            self.screen.blit(self.obj_img, obj_screen_pos)

        agent_screen_pos = (cur_loc[1] * self.TILE_SIZE + self.MARGIN,
                            cur_loc[0] * self.TILE_SIZE + self.MARGIN)
        self.screen.blit(self.agent_img, agent_screen_pos)

        for i, j, dir in self.env.walls:
            scr_x = j * self.TILE_SIZE
            scr_y = i * self.TILE_SIZE

            if dir == 'v':
                self.screen.blit(self.obj_fence_v,
                                 (scr_x - self.WALL_THICKNESS // 2, scr_y))
            elif dir == 'h':
                self.screen.blit(self.obj_fence_h,
                                 (scr_x, scr_y - self.WALL_THICKNESS // 2))

    def get_action_from_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return Action.UP
        elif keys[pygame.K_DOWN]:
            return Action.DOWN
        elif keys[pygame.K_LEFT]:
            return Action.LEFT
        elif keys[pygame.K_RIGHT]:
            return Action.RIGHT
        elif keys[pygame.K_SPACE]:
            return Action.PICKUP
        elif keys[pygame.K_RETURN]:
            return Action.STAY
        return None

    def run(self):
        accumulated_reward = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            state = self.env.get_current_state()
            if self.agent is None:
                action = self.get_action_from_keys()
            else:
                time.sleep(0.5)
                action = self.agent.get_best_action(state)

            if action:
                state, reward = self.env.step(action)
                accumulated_reward += reward

            if self.env.terminated():
                print("Episode finished. Accumulated reward: ",
                      accumulated_reward)
                pygame.time.wait(500)
                self.env.reset()
                accumulated_reward = 0

            self.draw_grid()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
        sys.exit()
