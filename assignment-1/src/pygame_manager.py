import pygame


def handle_events(event, config):
    match event.type:
        case pygame.QUIT:
            config["running"] = False
