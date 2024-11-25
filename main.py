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


def bpr():
    X, y = make_moons(n_samples=100, noise = 0.1, random_state=21)
    y = y.reshape((y.shape[0], 1)) #from a big array to a multiples little arrays

    nn = NeuralNet([2, 6, 6, 1])
    nn.setup_training(X, y)

    # yy = nn.forward_propagation(X)[-1] > 0.5

    pygame.init()

    width, height = 640, 480
    square_size = 300
    pixel_size = 5
    min_x = -3
    max_x = 3
    min_y = -3
    max_y = 3
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pixel Array Display")
    square_x, square_y = (width - square_size) // 2, (height - square_size) // 2
    grid_width = square_size // pixel_size
    grid_height = square_size // pixel_size
    x_coords = np.linspace(min_x, max_x, grid_width)
    y_coords = np.linspace(max_y, min_y, grid_height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    pixels_bg = np.column_stack((xx.ravel(), yy.ravel()))

    square_surface = pygame.Surface((square_size, square_size))

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # Font for displaying FPS

    nn.iteration_training(1000)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        square_surface.fill((0, 0, 0))

        #nn.iteration_training()
        pixels_bg_colors = nn.forward_propagation(pixels_bg)[-1]

        for gy in range(grid_height):
            for gx in range(grid_width):
                v = (pixels_bg_colors[gy * grid_width + gx])
                color = color_by_coordinates(v, (230, 126, 34), (52, 152, 219))
                pygame.draw.rect(square_surface, color,(gx * pixel_size, gy * pixel_size, pixel_size, pixel_size))

        my_y = nn.forward_propagation(X)[-1] > 0.5

        for p in range(X.shape[0]):
            px, py = X[p]
            py *= -1

            px = ((px - min_x) / (max_x - min_x)) * square_size
            py = ((py - min_y) / (max_y - min_y)) * square_size

            color = (255, 255, 255)

            if y[p] == 1:
                if my_y[p] == 1:
                    color = (0, 255, 0) #TP = GREEN
                else:
                    color = (255, 0, 255) #FN = PURPLE
            else:
                if my_y[p] == 1:
                    color = (255, 255, 0)  # FP = YELLOW
                else:
                    color = (255, 0, 0)  # TN = RED
            pygame.draw.circle(square_surface, color, (int(px), int(py)), 4)

        screen.fill((30, 30, 30))
        screen.blit(square_surface, (square_x, square_y))

        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def gna():
    # X, y = make_moons(n_samples=100, noise=0.1, random_state=21)
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=21)
    y = y.reshape((y.shape[0], 1))  # from a big array to a multiples little arrays

    ga = GeneticAlgorithms(X, y, [2, 16, 16, 1])

    pygame.init()

    width, height = 640, 480
    square_size = 300
    pixel_size = 10
    min_x = -2
    max_x = 2
    min_y = -2
    max_y = 2
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pixel Array Display")
    square_x, square_y = (width - square_size) // 2, (height - square_size) // 2
    grid_width = square_size // pixel_size
    grid_height = square_size // pixel_size
    x_coords = np.linspace(min_x, max_x, grid_width)
    y_coords = np.linspace(max_y, min_y, grid_height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    pixels_bg = np.column_stack((xx.ravel(), yy.ravel()))

    square_surface = pygame.Surface((square_size, square_size))

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)  # Font for displaying FPS

    ga.iterate(100)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        square_surface.fill((0, 0, 0))

        ga.iterate(1)
        pixels_bg_colors = ga.get_bestIndividualResult(pixels_bg)[-1]

        for gy in range(grid_height):
            for gx in range(grid_width):
                v = (pixels_bg_colors[gy * grid_width + gx])
                color = color_by_coordinates(v, (230, 126, 34), (52, 152, 219))
                pygame.draw.rect(square_surface, color, (gx * pixel_size, gy * pixel_size, pixel_size, pixel_size))

        my_y = ga.get_bestIndividualResult(X)[-1] > 0.5

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

        screen.fill((30, 30, 30))
        screen.blit(square_surface, (square_x, square_y))

        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        text = font.render(f"Best MSE: {ga.get_bestScore()}", True, (255, 255, 255))
        screen.blit(text, (80, 10))

        text = font.render(f"Best MSE this iteration: {ga.get_bestScoreThisIt()}", True, (255, 255, 255))
        screen.blit(text, (80, 40))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    gna()


if __name__ == '__main__':
    main()
