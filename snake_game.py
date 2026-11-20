import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Game constants
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
CELL_SIZE = 20
FPS = 10

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        start_x = WINDOW_WIDTH // 2
        start_y = WINDOW_HEIGHT // 2
        self.body = [
            (start_x, start_y),
            (start_x - CELL_SIZE, start_y),
            (start_x - 2 * CELL_SIZE, start_y),
        ]
        self.direction = RIGHT
        self.grow = False

    def move(self):
        head_x, head_y = self.body[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x * CELL_SIZE, head_y + dir_y * CELL_SIZE)
        self.body.insert(0, new_head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def change_direction(self, new_direction):
        # Prevent 180-degree turns
        opposite = (-self.direction[0], -self.direction[1])
        if new_direction != opposite:
            self.direction = new_direction

    def check_collision(self):
        head = self.body[0]
        # Wall collision
        if head[0] < 0 or head[0] >= WINDOW_WIDTH:
            return True
        if head[1] < 0 or head[1] >= WINDOW_HEIGHT:
            return True
        # Self collision
        if head in self.body[1:]:
            return True
        return False

    def draw(self, surface):
        for i, segment in enumerate(self.body):
            color = GREEN if i == 0 else DARK_GREEN
            rect = pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)


class Food:
    def __init__(self):
        self.position = (0, 0)
        self.spawn([])

    def spawn(self, snake_body):
        while True:
            x = random.randrange(0, WINDOW_WIDTH, CELL_SIZE)
            y = random.randrange(0, WINDOW_HEIGHT, CELL_SIZE)
            self.position = (x, y)
            if self.position not in snake_body:
                break

    def draw(self, surface):
        rect = pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, RED, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)


def draw_grid(surface):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, GRAY, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, GRAY, (0, y), (WINDOW_WIDTH, y))


def draw_score(surface, font, score):
    text = font.render(f"Score: {score}", True, WHITE)
    surface.blit(text, (10, 10))


def draw_game_over(surface, font, score):
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    surface.blit(overlay, (0, 0))

    go_text = font.render("GAME OVER", True, RED)
    score_text = font.render(f"Final Score: {score}", True, WHITE)
    hint_text = font.render("Press R to restart or Q to quit", True, WHITE)

    surface.blit(go_text, go_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40)))
    surface.blit(score_text, score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)))
    surface.blit(hint_text, hint_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40)))


def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    snake = Snake()
    food = Food()
    food.spawn(snake.body)
    score = 0
    game_over = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_r:
                        snake.reset()
                        food.spawn(snake.body)
                        score = 0
                        game_over = False
                    elif event.key == pygame.K_q:
                        running = False
                else:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        snake.change_direction(UP)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        snake.change_direction(DOWN)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        snake.change_direction(LEFT)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        snake.change_direction(RIGHT)

        if not game_over:
            snake.move()

            # Eat food
            if snake.body[0] == food.position:
                snake.grow = True
                score += 1
                food.spawn(snake.body)

            # Collision check
            if snake.check_collision():
                game_over = True

        # Drawing
        screen.fill(BLACK)
        draw_grid(screen)
        snake.draw(screen)
        food.draw(screen)
        draw_score(screen, font, score)

        if game_over:
            draw_game_over(screen, font, score)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()