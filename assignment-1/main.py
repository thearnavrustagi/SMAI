import json
import pygame

from game import SpaceJamm
import pygame_manager as pgm
import constants as c


screen = pygame.display.set_mode(c.GAME_SIZE)
config = json.load(open("pygame_config.json"))

if __name__ == "__main__":
    space_jam = SpaceJamm("./maps/map_1.txt")
    space_jam.show_board()
    while config["running"]:
        for event in pygame.event.get():
            pgm.handle_events(event, config)
        screen.fill(c.SPACE_BG)
        for blitable, coords in space_jam.get_blitables():
            screen.blit(blitable, coords)

        pygame.display.flip()

    pygame.quit()
