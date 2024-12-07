import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from nn import NeuralNet

from ga import GeneticAlgorithms


def color_by_coordinates(value, color1, color2):
        value = max(0, min(value, 1))

        color1 = [color1[0], color1[1], color1[2]]
        color2 = [color2[0], color2[1], color2[2]]

        if value <= 0.5:
            factor = value / 0.5
            r = color1[0] * (1 - factor) + 255 * factor
            g = color1[1] * (1 - factor) + 255 * factor
            b = color1[2] * (1 - factor) + 255 * factor
        else:
            factor = (value - 0.5) / 0.5
            r = 255 * (1 - factor) + color2[0] * factor
            g = 255 * (1 - factor) + color2[1] * factor
            b = 255 * (1 - factor) + color2[2] * factor

        return (r, g, b)


def setup_screen(height, max_x, max_y, min_x, min_y, pixel_size, square_size, width):
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NN experimentation")
    square_x, square_y = (width - square_size) // 2, (height - square_size) // 2
    grid_width = square_size // pixel_size
    grid_height = square_size // pixel_size
    x_coords = np.linspace(min_x, max_x, grid_width)
    y_coords = np.linspace(max_y, min_y, grid_height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    pixels_bg = np.column_stack((xx.ravel(), yy.ravel()))
    square_surface1 = pygame.Surface((square_size, square_size))
    square_surface2 = pygame.Surface((square_size, square_size))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # Font for displaying FPS
    return clock, font, grid_height, grid_width, pixels_bg, screen, square_surface1, square_surface2, square_x, square_y


def draw_datasets(X, y, max_x, max_y, min_x, min_y, my_y, square_size, square_surface):
    for p in range(X.shape[0]):
        px, py = X[p]
        py *= -1

        px = ((px - min_x) / (max_x - min_x)) * square_size
        py = ((py - min_y) / (max_y - min_y)) * square_size

        color = (255, 255, 255)

        if y[p] == 1:
            if my_y[p] == 1:
                color = (0, 255, 0)  # TP = GREEN
            else:
                color = (255, 0, 255)  # FN = PURPLE
        else:
            if my_y[p] == 1:
                color = (255, 255, 0)  # FP = YELLOW
            else:
                color = (255, 0, 0)  # TN = RED
        pygame.draw.circle(square_surface, color, (int(px), int(py)), 4)


def get_confusion_matrix(y, yy):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for p in range(y.shape[0]):
        if y[p] == 1:
            if yy[p] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if yy[p] == 1:
                FP += 1
            else:
                TN += 1
    return TP, FN, FP, TN


def draw_background(grid_height, grid_width, pixel_size, square_surface, pixels_bg_colors):
    for gy in range(grid_height):
        for gx in range(grid_width):
            v = (pixels_bg_colors[gy * grid_width + gx])
            color = color_by_coordinates(v, (230, 126, 34), (52, 152, 219))
            pygame.draw.rect(square_surface, color, (gx * pixel_size, gy * pixel_size, pixel_size, pixel_size))


def main():
    X, y = make_moons(n_samples=100, noise = 0.1, random_state=21)
    y = y.reshape((y.shape[0], 1)) #from a big array to a multiples little arrays

    nn = NeuralNet([2, 6, 6, 1])
    nn.setup_training(X, y)
    nn.iteration_training(10)

    ga = GeneticAlgorithms(X, y, [2, 16, 16, 1], mode=1)
    ga.iterate(100)

    # yy = nn.forward_propagation(X)[-1] > 0.5

    pygame.init()

    width, height = 960, 480
    square_size = 300
    pixel_size = 10
    min_x = -3
    max_x = 3
    min_y = -3
    max_y = 3
    clock, font, grid_height, grid_width, pixels_bg, screen, square_surface1, square_surface2, square_x, square_y = setup_screen(height,
                                                                                                               max_x,
                                                                                                               max_y,
                                                                                                               min_x,
                                                                                                               min_y,
                                                                                                               pixel_size,
                                                                                                               square_size,
                                                                                                               width)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        square_surface1.fill((0, 0, 0))
        square_surface2.fill((0, 0, 0))

        nn.iteration_training() #nn
        pixels_bg_colors = nn.forward_propagation(pixels_bg)[-1]
        draw_background(grid_height, grid_width, pixel_size, square_surface1, pixels_bg_colors)

        ga.iterate(1) # ga
        pixels_bg_colors = ga.get_bestIndividualResult(pixels_bg)[-1]
        draw_background(grid_height, grid_width, pixel_size, square_surface2, pixels_bg_colors)


        my_y = nn.forward_propagation(X)[-1] > 0.5
        draw_datasets(X, y, max_x, max_y, min_x, min_y, my_y, square_size, square_surface1)

        my_y = ga.get_bestIndividualResult(X)[-1] > 0.5
        draw_datasets(X, y, max_x, max_y, min_x, min_y, my_y, square_size, square_surface2)

        screen.fill((30, 30, 30))
        screen.blit(square_surface1, (square_x - 200, square_y))
        screen.blit(square_surface2, (square_x + 200, square_y))

        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        text = font.render(f"Best MSE: {ga.get_bestScore()}", True, (255, 255, 255))
        screen.blit(text, (500, 10))

        text = font.render(f"Best MSE this iteration: {ga.get_bestScoreThisIt()}", True, (255, 255, 255))
        screen.blit(text, (500, 30))

        text = font.render(f"Mean MSE this iteration: {ga.get_meanScoreThisIt()}", True, (255, 255, 255))
        screen.blit(text, (500, 50))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
