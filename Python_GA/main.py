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
        self.simulation = False
        self.new_game()

    def draw_grid(self):
        [pg.draw.line(self.screen, [50] * 3, (x, 0), (x, self.WINDOW_SIZE)) for x in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]
        [pg.draw.line(self.screen, [50] * 3, (0, y), (self.WINDOW_SIZE, y)) for y in range(0, self.WINDOW_SIZE, self.TILE_SIZE)]

    def new_game(self):
        if self.simulation == False:
            # initialize the initial population
            self.population = Population(self.POPULATION_SIZE, self)
            self.genetic_algorithm = Genetic_algorithm(100, self)
        else:
            saved_individual_file = open("saved_individuals/best_individual_V20230524020042.pkl", 'rb')
            brain = pickle.load(saved_individual_file)
            self.individual = Snake(self, brain, True)

    def update(self):
        if self.simulation == False:
            self.population.update()
            if self.population.all_done:
                self.population = self.genetic_algorithm.next_generation(self.population)
        else:
            self.individual.update()

        pg.display.flip()
        self.clock.tick(60)

    def draw(self):
        self.screen.fill("black")
        if self.simulation == False:
            if self.show_individuals:
                self.draw_grid()
                self.population.draw()
        else:
            self.draw_grid()
            self.individual.draw(255)

    def check_event(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                if self.simulation == False:
                    self.genetic_algorithm.save_best_individual()
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