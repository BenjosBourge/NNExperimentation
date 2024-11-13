import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score
from nn import NeuralNet


def color_by_coordinates(v):
    v = np.clip(v, -1, 1)
    red = 255 * max(v, 0)
    green = -255 * min(v, 0)
    blue = 128 * max(green / 255, red / 255)
    return 255 - red, 255 - green, 255 - blue


def main():
    X, y = make_moons(n_samples=100, noise = 0.1, random_state=21)
    y = y.reshape((y.shape[0], 1)) #from a big array to a multiples little arrays

    print(X)

    nn = NeuralNet([6, 6, 1])
    nn.setup_training(X, y)

    # yy = nn.forward_propagation(X)[-1] > 0.5

    pygame.init()

    width, height = 640, 480
    square_size = 300
    pixel_size = 5
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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        square_surface.fill((0, 0, 0))

        nn.iteration_training()
        pixels_bg_colors = nn.forward_propagation(pixels_bg)[-1]

        for y in range(grid_height):
            for x in range(grid_width):
                v = (pixels_bg_colors[y * grid_width + x]) * 2 - 1
                color = color_by_coordinates(v)
                pygame.draw.rect(square_surface, color,(x * pixel_size, y * pixel_size, pixel_size, pixel_size))

        for p in range(X.shape[0]):
            px, py = X[p]
            py *= -1

            px = ((px - min_x) / (max_x - min_x)) * square_size
            py = ((py - min_y) / (max_y - min_y)) * square_size

            pygame.draw.circle(square_surface, (255, 0, 0), (int(px), int(py)), 2)

        px = 0
        py = 0

        px = ((px - min_x) / (max_x - min_x)) * square_size
        py = ((py - min_y) / (max_y - min_y)) * square_size

        pygame.draw.circle(square_surface, (255, 255, 0), (int(px), int(py)), 4)

        screen.fill((30, 30, 30))
        screen.blit(square_surface, (square_x, square_y))

        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
