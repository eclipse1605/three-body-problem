import pygame
import sys
import matplotlib.pyplot as plt
import argparse
import os
import visualization
import utils
from system import System
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="3-Body Problem Simulation with Lyapunov Exponent Analysis"
    )
    parser.add_argument(
        "--map",
        type=str,
        default="data/fullMap.csv",
        help="Path to the heatmap data file",
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="Run in headless mode (no UI)"
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    pygame.init()
    pygame.display.set_caption("N-Body Simulation - Press 'u' to create animation")
    clock = pygame.time.Clock()

    print("\n3-Body Problem Simulation Controls:")
    print("----------------------------------")
    print("- Press 'i' to enhance (generate higher resolution data for current view)")
    print("- Press 'u' at any point to create an animation of that initial condition")
    print("- Press 'n' to loop the animation at the point of closest proximity")
    print("- Press 'z' to automatically zoom to minimum proximity point\n")

    if not os.path.exists(args.map):
        print(f"Map file {args .map } not found. Generating default map...")
        utils.run("0.0 1.0 0.0 1.0", 50)
        args.map = "data/zoom.csv"

    plot = visualization.plot_proximity_heatmap(args.map)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        clock.tick(30)


if __name__ == "__main__":
    main()
