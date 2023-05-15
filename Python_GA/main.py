import pygame as pg
from game_objects import *
from genetic_algorithm import *
import sys

class Game:
    def __init__(self):
        pg.init()
        self.WINDOW_SIZE = 600
        self.TILE_SIZE = 20
        self.screen = pg.display.set_mode([self.WINDOW_SIZE] * 2)
        pg.display.set_caption("Snake game AI - genetic algorithm")
        self.clock = pg.time.Clock()
        self.POPULATION_SIZE = 50
        self.current_snake_index = 0
        self.show_individuals = True
        self.new_game()

    def draw_grid(self):
        [pg.draw.line(self.screen, [50] * 3, (x, 0), (x, self.WINDOW_SIZE)) for x in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]
        [pg.draw.line(self.screen, [50] * 3, (0, y), (self.WINDOW_SIZE, y)) for y in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]

    def new_game(self):
        # initialize the initial population
        self.population = Population(self.POPULATION_SIZE, self)
        self.genetic_algorithm = Genetic_algorithm(100, self)

    def update(self):
        # TODO update genetic algorithm
        self.population.update()
        if self.population.all_done:
            # TODO next generation
            self.population = self.genetic_algorithm.next_generation(self.population)

        pg.display.flip()
        self.clock.tick(60)

    def draw(self):
        self.screen.fill("black")
        self.draw_grid()
        if self.show_individuals:
            self.population.draw()
        # TODO draw the algorithm (the snakes in the current population)

    def check_event(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            # snake control
            #self.population.get_current_individual().control(event)
            # DEBUGGING
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    self.show_individuals = not self.show_individuals

    def run(self):
        while True:
            self.check_event()
            self.update()
            self.draw()

if __name__ == '__main__':
    game = Game()
    game.run()