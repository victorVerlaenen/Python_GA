import pygame as pg
from random import randrange
import random
from neural_network import *

vec2 = pg.math.Vector2

class Snake:
    def __init__(self, game, brain, simulate = False):
        #print("New snake")
        self.game = game
        self.size = game.TILE_SIZE
        self.rect = pg.rect.Rect([0, 0, game.TILE_SIZE, game.TILE_SIZE])
        if simulate:
            self.step_delay = 20
        else:
            self.step_delay = 0 #milliseconds
        self.time = 0
        self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}
        if brain == None:
            self.brain = Feedforward_neural_network(9, 3, 2, 15)
        else:
            self.brain = brain
        self.inputs = []
        # add a fitness function and the associated values needed
        # add the amount of steps per snake
        self.STEPS_TO_TAKE = 5000
        self.STEPS_FOR_PENALTY = 200
        self.penalties = 0
        self.steps_taken = 0
        self.score = 0
        self.record = 0
        self.length = 1
        self.average_steps = 0
        self.deaths = -1 # initialize to -1 because the first respawn is not a death
        self.respawn()

    def respawn(self):
        self.check_record()
        self.rect.center = self.get_random_position()
        self.direction = self.get_random_direction()
        self.penalty_steps = 0
        self.segments = []
        self.length = 1
        self.deaths += 1
        self.food = Food(self.game, self)

    def check_record(self):
        if (self.length - 1) > self.record:
            self.record = (self.length - 1)

    def control(self, event):
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_w and self.directions[pg.K_w]:
                self.direction = vec2(0, -self.size)
                self.directions = {pg.K_w: 1, pg.K_s: 0, pg.K_a: 1, pg.K_d: 1}
            if event.key == pg.K_s and self.directions[pg.K_s]:
                self.direction = vec2(0, self.size)
                self.directions = {pg.K_w: 0, pg.K_s: 1, pg.K_a: 1, pg.K_d: 1}
            if event.key == pg.K_a and self.directions[pg.K_a]:
                self.direction = vec2(-self.size, 0)
                self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 1, pg.K_d: 0}
            if event.key == pg.K_d and self.directions[pg.K_d]:
                self.direction = vec2(self.size, 0)
                self.directions = {pg.K_w: 1, pg.K_s: 1, pg.K_a: 0, pg.K_d: 1}

    def calculate_fitness(self):
        self.calculate_average_steps()
        self.fitness = (self.record * 5000) - (self.deaths * 150) - (self.average_steps * 100) - (self.penalties * 1500)
        return self.fitness

    def delta_time(self):
        time_now = pg.time.get_ticks()
        if time_now - self.time > self.step_delay:
            self.time = time_now
            return True
        return False

    def get_random_position(self):
        x = randrange(self.size // 2, self.game.WINDOW_SIZE - self.size // 2, self.size)
        y = randrange(self.size // 2, self.game.WINDOW_SIZE - self.size // 2, self.size)
        return [x, y]

    def get_random_direction(self):
        directions = [vec2(self.size, 0), vec2(-self.size, 0), vec2(0, self.size), vec2(0, -self.size)]
        return random.choice(directions)

    def check_for_movement_penalty(self):
        if self.penalty_steps >= self.STEPS_FOR_PENALTY:
            self.penalties += 1
            self.respawn()

    def calculate_average_steps(self):
        if self.score != 0:
            self.average_steps = self.average_steps/self.score

    def check_if_done(self):
        if self.steps_taken >= self.STEPS_TO_TAKE:
            return True
        return False

    def check_border(self):
        if self.rect.left < 0 or self.rect.right > self.game.WINDOW_SIZE:
            self.respawn()
        if self.rect.top < 0 or self.rect.bottom > self.game.WINDOW_SIZE:
            self.respawn()

    def check_food(self):
        if self.rect.center == self.food.rect.center:
            self.food.rect.center = self.get_random_position()
            self.length += 1
            self.score += 1
            self.average_steps += self.penalty_steps
            self.penalty_steps = 0

    def check_selfeating(self):
        if len(self.segments) != len(set(segment.center for segment in self.segments)):
            self.respawn()

    def calculate_inputs(self):
        def set_input(idx, val):
            self.inputs[idx] = val

        # Initialy set all inputs to 0
        self.inputs = [0] * 9  

        # Which direction is the snake moving
        if self.direction.x > 0:
            set_input(0,1)
        if self.direction.x < 0:
            set_input(1,1)
        if self.direction.y > 0:
            set_input(2,1)
        if self.direction.y < 0:
            set_input(3,1)

        # Check for danger to either side of the snakes head
        half_size_of_segment = (self.size/2)
        if self.rect.centerx >= self.game.WINDOW_SIZE - half_size_of_segment or self.is_segment_right():
            set_input(4, 1)
        if self.rect.centerx <= half_size_of_segment or self.is_segment_left():
            set_input(5, 1)
        if self.rect.centery >= self.game.WINDOW_SIZE - half_size_of_segment or self.is_segment_bottom():
            set_input(6, 1)
        if self.rect.centery <= half_size_of_segment or self.is_segment_top():
            set_input(7, 1)
        
        # Value of sinus of angle on which the food is inclined relative to the snake
        snake_to_food_vec = vec2(self.food.rect.center) - vec2(self.rect.center)
        if np.linalg.norm(snake_to_food_vec) == 0:
            angle_between = 0
        else:
            v1 = np.array([self.direction.x, self.direction.y, 0])
            v2 = np.array([snake_to_food_vec.x, snake_to_food_vec.y, 0])
            direction_normalized = v1 / np.linalg.norm(v1)
            snake_to_food_normalized = v2 / np.linalg.norm(v2)

            dot_product = np.dot(direction_normalized, snake_to_food_normalized)
            angle_between = np.arccos(dot_product)

            cross_product = np.cross(direction_normalized, snake_to_food_normalized)
            if cross_product[2] < 0:
                angle_between = -angle_between

        sin_of_angle = np.sin(angle_between)
        set_input(8, sin_of_angle)

    def normalize_inputs(self):    
        means = np.mean(self.inputs, axis = 0)
        stds = np.std(self.inputs, axis=0)
        self.inputs = (self.inputs - means) / stds

    def is_segment_left(self):
        head_x = self.rect.centerx
        for segment in self.segments[1:]:
            if segment.centerx == head_x - self.size:
                return True
        return False

    def is_segment_top(self):
        head_y = self.rect.centery
        for segment in self.segments[1:]:
            if segment.centery == head_y - self.size:
                return True
        return False

    def is_segment_right(self):
        head_x = self.rect.centerx
        for segment in self.segments[1:]:
            if segment.centerx == head_x + self.size:
                return True
        return False

    def is_segment_bottom(self):
        head_y = self.rect.centery
        for segment in self.segments[1:]:
            if segment.centery == head_y + self.size:
                return True
        return False

    def move(self):
        if self.delta_time():
            self.interpret_brain_output(self.inputs)
            self.rect.move_ip(self.direction)
            self.steps_taken += 1
            self.penalty_steps += 1
            self.segments.append(self.rect.copy())
            self.segments = self.segments[-self.length:]

    def interpret_brain_output(self, inputs):
        output = self.brain.forward(inputs)
        highest_index = np.argmax(output)
        if highest_index == 0:
            # turn left
            self.direction.rotate_ip(90)
        if highest_index == 1:
            # keep going straight
            pass
        if highest_index == 2:
            # turn right
            self.direction.rotate_ip(-90)

    def update(self):
        self.check_for_movement_penalty()
        self.check_selfeating()
        self.check_border()
        self.check_food()
        self.calculate_inputs()
        self.move()

    def draw(self, opacity):
        surface = pg.Surface((self.game.WINDOW_SIZE, self.game.WINDOW_SIZE), pg.SRCALPHA)
        green = (0, 255, 0, opacity)
        [pg.draw.rect(surface, green, segment) for segment in self.segments] 
        self.food.draw(opacity, surface)

class Food:
    def __init__(self, game, snake):
        #print("New food")
        self.game = game
        self.size = game.TILE_SIZE
        self.rect = pg.rect.Rect([0, 0, game.TILE_SIZE, game.TILE_SIZE])
        self.rect.center = snake.get_random_position()

    def draw(self, opacity, surface):
        red = (255, 0, 0, opacity)
        pg.draw.rect(surface, red, self.rect)
        self.game.screen.blit(surface, (0,0))