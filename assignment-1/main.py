import json
import pygame

from spacejamm import SpaceJamm
import pygame_manager as pgm
import constants as c


screen = pygame.display.set_mode(c.GAME_SIZE)
config = json.load(open("pygame_config.json"))

if __name__ == "__main__":
    space_jam = SpaceJamm("./maps/map_1.txt", screen)
    while config["running"]:
        for event in pygame.event.get():
            pgm.handle_events(event, config)

        space_jam.depth_first_search()

    pygame.quit()
