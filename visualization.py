import pygame
import matplotlib.pyplot as plt
from system import System
from matplotlib.animation import FuncAnimation
import numpy as np
import csv
import utils
import pandas as pd
import os
from pathlib import Path


def plot_proximity_heatmap(data_path, next=False):
    data_path = os.path.normpath(data_path)
    data = pd.read_csv(data_path)
    vx_values = data["vx"].values
    vy_values = data["vy"].values
    proximity_values = data["proximity"].values

    vx_unique = np.unique(vx_values)
    vy_unique = np.unique(vy_values)
    num_vx = len(vx_unique)
    num_vy = len(vy_unique)

    # Debug information
    print(f"Data size: {len(proximity_values)}")
    print(f"Grid dimensions: {num_vy}x{num_vx} = {num_vy * num_vx}")

    # Handle reshaping properly
    if num_vx * num_vy != len(proximity_values):
        print(
            f"Warning: Grid dimensions ({num_vy}x{num_vx}={num_vy*num_vx}) don't match data size ({len(proximity_values)})"
        )
        # Use square dimensions if needed
        grid_size = int(np.sqrt(len(proximity_values)))
        print(f"Using {grid_size}x{grid_size} grid instead")
        proximity_heatmap = proximity_values[: grid_size * grid_size].reshape(
            grid_size, grid_size
        )
        # Adjust unique values for correct plot extents
        if len(vx_unique) > grid_size:
            vx_unique = np.linspace(vx_unique[0], vx_unique[-1], grid_size)
        if len(vy_unique) > grid_size:
            vy_unique = np.linspace(vy_unique[0], vy_unique[-1], grid_size)
    else:
        proximity_heatmap = proximity_values.reshape(num_vy, num_vx)

    threshold = 3.5

    proximity_heatmap[proximity_heatmap >= threshold] = np.nan

    vx_range = np.unique(vx_values)
    vy_range = np.unique(vy_values)

    plt.figure(figsize=(12, 10), facecolor="#121212")
    plt.rcParams.update(
        {
            "text.color": "white",
            "axes.facecolor": "#1e1e1e",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "grid.color": "#444444",
            "figure.facecolor": "#121212",
            "savefig.facecolor": "#121212",
        }
    )

    extent = [vx_range.min(), vx_range.max(), vy_range.min(), vy_range.max()]

    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0, "midnightblue"),
        (0.25, "blue"),
        (0.5, "cyan"),
        (0.75, "yellow"),
        (1, "red"),
    ]
    custom_cmap = LinearSegmentedColormap.from_list("chaos_map", colors)
    custom_cmap.set_under("#3d0066")

    plt.imshow(
        proximity_heatmap,
        origin="lower",
        cmap=custom_cmap,
        extent=extent,
        aspect="auto",
        vmin=0.002,
        interpolation="gaussian",
    )

    cbar = plt.colorbar(label="Proximity", pad=0.02)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    plt.xlabel("Initial X Velocity (vx)", fontsize=12, fontweight="bold")
    plt.ylabel("Initial Y Velocity (vy)", fontsize=12, fontweight="bold")
    plt.title(
        "Chaos in 3-Body Problem: Proximity Mapping",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.grid(color="#444444", linestyle="--", linewidth=0.5, alpha=0.7)

    vx_selected = None
    vy_selected = None

    def on_key(event):
        nonlocal vx_selected, vy_selected
        if event.key == "u":
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if (
                    vx_range.min() <= x <= vx_range.max()
                    and vy_range.min() <= y <= vy_range.max()
                ):
                    x_index = int(
                        (x - vx_range.min())
                        / (vx_range.max() - vx_range.min())
                        * (num_vx)
                    )
                    y_index = int(
                        (y - vy_range.min())
                        / (vy_range.max() - vy_range.min())
                        * (num_vy)
                    )
                    vx_selected = vx_range[x_index]
                    vy_selected = vy_range[y_index]
                    print(
                        f"Pressed u at (vx, vy): {vx_selected }, {vy_selected }, {proximity_heatmap [y_index ,x_index ]}"
                    )
                    print("Generating animation with:", vx_selected, vy_selected)
                    str_arr = " ".join(map(str, [1, vx_selected, vy_selected]))
                    utils.get_positions(str_arr)
                    pygame_animate(os.path.join("data", "positions.csv"))
        if event.key == "m":
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(f"Coordinates: x={x :.5f}, y={y :.5f}")
        if event.key == "h":
            plt.axis([0, 1, 0, 1])
            plt.draw()
        if event.key == "n":
            print("Creating loop animation")
            utils.loop_csv()
            pygame_animate(os.path.join("data", "cut_positions.csv"))
        if event.key == "i":

            xLim = plt.gca().get_xlim()
            yLim = plt.gca().get_ylim()

            plt.close()

            print("\nEnhancing current view with higher resolution...")
            print(f"  vx range: [{xLim [0 ]:.5f}, {xLim [1 ]:.5f}]")
            print(f"  vy range: [{yLim [0 ]:.5f}, {yLim [1 ]:.5f}]")

            new_ax = enhance(xLim, yLim)

            plt.figure()
            plt.sca(new_ax)
            plt.draw()
            plt.show(block=False)
        if event.key == "z":
            xLim = plt.gca().get_xlim()
            yLim = plt.gca().get_ylim()
            x, y = find_minimum_proximity(os.path.join("data", "zoom.csv"), xLim, yLim)
            zoom_factor = abs(xLim[0] - xLim[1]) / 4
            zoom(x, y, zoom_factor)
            plt.draw()
        if event.key == "a":
            print("Auto-zoom functionality has been removed")

    ax = plt.gca()
    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    plt.figtext(
        0.01,
        0.01,
        "Keys: i=enhance, u=animate, n=loop, z=zoom to min",
        fontsize=9,
        color="gray",
    )

    if next:
        plt.show(block=False)
    else:
        plt.show()
    return ax


def zoom(x, y, factor):
    plt.axis([x - factor, x + factor, y - factor, y + factor])


def find_minimum_proximity(csv_file, xLim, yLim):
    try:

        csv_file = os.path.normpath(csv_file)

        df = pd.read_csv(csv_file)

        df_filtered = df[
            (df["vx"] >= xLim[0])
            & (df["vx"] <= xLim[1])
            & (df["vy"] >= yLim[0])
            & (df["vy"] <= yLim[1])
        ]

        if df_filtered.empty:
            print("No data points found in the specified range.")

            return (xLim[0] + xLim[1]) / 2, (yLim[0] + yLim[1]) / 2

        min_proximity_row = df_filtered.loc[df_filtered["proximity"].idxmin()]
        min_proximity = df_filtered["proximity"].min()

        x, y = min_proximity_row["vx"], min_proximity_row["vy"]

        print(
            f"Found minimum proximity point at (vx={x :.5f}, vy={y :.5f}) with value {min_proximity :.6f}"
        )
        return x, y

    except Exception as e:
        print(f"Error in find_minimum_proximity: {e }")

        return (xLim[0] + xLim[1]) / 2, (yLim[0] + yLim[1]) / 2


def enhance(xLim, yLim):
    try:

        input1 = f"{xLim [0 ]} {xLim [1 ]} {yLim [0 ]} {yLim [1 ]}"
        resolution = 50

        print(f"Generating {resolution }x{resolution } resolution map for region:")
        print(f"  vx range: [{xLim [0 ]:.5f}, {xLim [1 ]:.5f}]")
        print(f"  vy range: [{yLim [0 ]:.5f}, {yLim [1 ]:.5f}]")

        utils.run(input1, resolution)

        zoom_csv_path = os.path.join("data", "zoom.csv")

        print("Rendering new proximity map...")
        ax = plot_proximity_heatmap(zoom_csv_path, next=True)

        return ax

    except Exception as e:
        print(f"Error in enhance: {e }")

        return plt.gca()


def get_axis_limits(ax):
    return ax.get_xlim(), ax.get_ylim()


def read_csv(filename, num_bodies):
    positions = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        frames = []
        for row in reader:
            if row:
                frames.append([float(val) for val in row])
            else:
                if frames:
                    positions.append(frames)
                    frames = []
        if frames:
            positions.append(frames)
    positions = np.array(positions)

    return positions


def pygame_animate(positions_path):
    num_bodies = 3
    positions = read_csv(positions_path, num_bodies)
    if not pygame.get_init():
        pygame.init()

    system = System(state=np.zeros(12))
    win_size = (1200, 900)
    win = pygame.display.set_mode(win_size)
    pygame.display.set_caption(
        "N-Body Simulation - Press 'o' to exit, 't' to toggle trails"
    )

    font = pygame.font.SysFont("Arial", 18)

    body_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]

    max_trail_length = 100
    trails = [[] for _ in range(num_bodies)]
    show_trails = True

    print(f"Animation loaded with {positions .shape [0 ]} frames")
    print("Controls:")
    print("  'o' - Exit animation")
    print("  't' - Toggle motion trails")

    stars = []
    for _ in range(200):
        x = np.random.randint(0, win_size[0])
        y = np.random.randint(0, win_size[1])
        brightness = np.random.randint(100, 255)
        stars.append((x, y, brightness))

    frame_count = 0
    running = True
    while running and frame_count < positions.shape[0] * 10:

        win.fill((5, 5, 15))

        for star in stars:
            x, y, brightness = star
            color = (brightness, brightness, brightness)
            pygame.draw.circle(win, color, (x, y), 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    running = False
                if event.key == pygame.K_t:
                    show_trails = not show_trails

        current_frame = frame_count % positions.shape[0]

        center_x, center_y = win_size[0] // 2, win_size[1] // 2
        scale_factor = 200

        if show_trails:
            for body_idx in range(num_bodies):
                color = body_colors[body_idx]

                for i, pos in enumerate(trails[body_idx]):
                    alpha = (
                        int(255 * (i / len(trails[body_idx])))
                        if trails[body_idx]
                        else 0
                    )
                    trail_color = (
                        max(color[0] - (255 - alpha), 0),
                        max(color[1] - (255 - alpha), 0),
                        max(color[2] - (255 - alpha), 0),
                    )
                    pygame.draw.circle(
                        win, trail_color, pos, max(1, 3 * i / len(trails[body_idx]))
                    )

        for body_idx in range(num_bodies):
            if current_frame < len(positions) and body_idx < len(
                positions[current_frame]
            ):

                pos_x = positions[current_frame][body_idx][0] * scale_factor + center_x
                pos_y = positions[current_frame][body_idx][1] * scale_factor + center_y

                if show_trails:
                    trails[body_idx].append((pos_x, pos_y))
                    if len(trails[body_idx]) > max_trail_length:
                        trails[body_idx].pop(0)

                for radius in range(12, 4, -2):
                    alpha = 100 - (radius * 8)
                    if alpha > 0:
                        glow_color = (
                            min(body_colors[body_idx][0], alpha),
                            min(body_colors[body_idx][1], alpha),
                            min(body_colors[body_idx][2], alpha),
                        )
                        pygame.draw.circle(
                            win, glow_color, (int(pos_x), int(pos_y)), radius
                        )

                pygame.draw.circle(
                    win, body_colors[body_idx], (int(pos_x), int(pos_y)), 5
                )

        info_text = font.render(
            f"Frame: {current_frame }/{positions .shape [0 ]-1 } | Trails: {'On'if show_trails else 'Off'}",
            True,
            (200, 200, 200),
        )
        win.blit(info_text, (10, 10))

        pygame.display.flip()
        pygame.time.delay(20)
        frame_count += 1

    pygame.quit()


def plot_energy(delta_energy):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(delta_energy)), delta_energy)
    plt.xlabel("Time Step")
    plt.ylabel("Energy Difference")
    plt.title("Energy Conservation Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_proximity_heatmap(os.path.join("data", "zoom.csv"))
