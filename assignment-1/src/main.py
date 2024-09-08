import os
from argparse import ArgumentParser
import json
import pygame
from time import sleep

from board import Board
from spacejamm import SpaceJamm
import pygame_manager as pgm
import constants as c


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
            prog="SpaceJamm",
            description="State space search for the game Space Jamm"
            )
    parser.add_argument('-s', '--sim', action='store_true')
    parser.add_argument('-p', '--path')
    parser.add_argument('--search', choices=['dfs', 'bfs', 'best'], help='choose the type of search')
    parser.add_argument('--mapgen', type=int, help="give map_size and n_obs", nargs=2)
    parser.add_argument('-d', '--disp', action='store_true')

    return parser

def simulate(game, trail=[]):
    config = json.load(open("pygame_config.json"))

    while config["running"]:
        for event in pygame.event.get():
            pgm.handle_events(event, config)

        game.blit_board()
        if trail:
            game.animate(trail)
        sleep(5)

    pygame.quit()
    
def main():
    args = get_parser().parse_args()

    if args.sim and os.path.exists(str(args.path)):
        screen = pygame.display.set_mode(c.GAME_SIZE)
        space_jam = SpaceJamm(args.path, screen)
        
        match args.search:
            case "dfs":
                trail = space_jam.depth_first_search()
            case "bfs":
                trail = space_jam.breadth_first_search()
            case "best":
                trail = space_jam.best_first_search()
            case _:
                trail = space_jam.breadth_first_search()

        simulate(space_jam, trail)
    
    if args.mapgen:
        map_size, n_obs = args.mapgen
        board = Board()
        m = board.mapgen(int(map_size), int(n_obs))
        with open("../maps/test_map.txt", "w") as f:
            f.write(board.dump(m))
        print(board.dump(m))

    if args.disp and os.path.exists(str(args.path)):
        screen = pygame.display.set_mode(c.GAME_SIZE)
        space_jam = SpaceJamm(args.path, screen)

        simulate(space_jam)
    elif args.disp:
        screen = pygame.display.set_mode(c.GAME_SIZE)
        space_jam = SpaceJamm("../maps/test_map.txt", screen)

        simulate(space_jam)
        

if __name__ == "__main__":
    main()
