import numpy as np
import time
from functools import wraps
import pandas as pd
import subprocess
import csv
import os
import sys
from pathlib import Path


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func .__name__ } took {end_time -start_time :.4f} seconds to execute.")
        return result

    return wrapper


def make_state(d, vx, vy, bodies=3):
    middle_state = [0, 0, vx, vy]

    left = [d, 0, -middle_state[2] / 2, -middle_state[3] / 2]

    right = [-d, 0, -middle_state[2] / 2, -middle_state[3] / 2]
    return np.array(middle_state + left + right)


def rearrange(file_path):
    try:

        df = pd.read_csv(file_path)

        df_sorted = df.sort_values(by=["vy", "vx"])

        df_sorted.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error in rearrange(): {e }")
    else:
        print(f"Successfully rearranged data in {file_path }")


def run(input2, dimensions):
    try:

        os.makedirs("data", exist_ok=True)

        input1 = str("normal" + "\n")
        input2 = str(input2 + "\n")
        input3 = str(dimensions)

        exe_name = "lol.exe"
        if sys.platform != "win32":
            exe_name = "./lol"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        exe_path = os.path.join(current_dir, exe_name)

        if not os.path.exists(exe_path):
            print(
                f"Error: {exe_path } not found! Make sure to compile the C++ code first."
            )
            return

        process = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        process.stdin.write(input1.encode())
        process.stdin.write(input2.encode())
        process.stdin.write(input3.encode())

        output, error = process.communicate()

        if process.returncode == 0:
            print("Program ran successfully")

            rearrange(os.path.join("data", "zoom.csv"))
        else:
            print("Error running program:", error.decode())

        print("Output:", output.decode())
    except Exception as e:
        print(f"Error in run(): {e }")


def get_positions(state):
    try:

        os.makedirs("data", exist_ok=True)

        input1 = str("positions" + "\n")
        input2 = str(state)

        exe_name = "lol.exe"
        if sys.platform != "win32":
            exe_name = "./lol"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        exe_path = os.path.join(current_dir, exe_name)

        if not os.path.exists(exe_path):
            print(
                f"Error: {exe_path } not found! Make sure to compile the C++ code first."
            )
            return

        process = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        process.stdin.write(input1.encode())
        process.stdin.write(input2.encode())

        output, error = process.communicate()

        if process.returncode == 0:
            print("Program ran successfully")
        else:
            print("Error running program:", error.decode())

        print("Output:", output.decode())
    except Exception as e:
        print(f"Error in get_positions(): {e }")


def proximity(positions):
    initial_state = positions[0].flatten()
    min_distance = float("inf")
    min_step = 0
    di = 0

    for i, frame in enumerate(positions):
        state = frame.flatten()
        distance = np.linalg.norm(state - initial_state)
        if distance < min_distance and distance < di:
            min_distance = distance
            min_step = i
        di = distance

    print(f"Found minimum proximity at step {min_step } with distance {min_distance }")
    return min_step


def cut_csv(filename, stop_row):
    try:
        output_file = os.path.join("data", "cut_positions.csv")

        with open(filename, "r") as file:
            reader = csv.reader(file)
            rows = []
            row_num = 0
            for row in reader:
                if row:
                    rows.append(row)
                    row_num += 1
                else:
                    row_num += 1
                if row_num > stop_row * 3 + stop_row - 1:
                    break

        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            for i in range(stop_row):
                for j in range(3):
                    writer.writerow(rows[i * 3 + j])
                writer.writerow([])
    except Exception as e:
        print(f"Error in cut_csv(): {e }")


def loop_csv():
    import visualization

    positions_file = os.path.join("data", "positions.csv")

    try:
        positions = visualization.read_csv(positions_file, 3)
        min_step = proximity(positions)
        cut_csv(positions_file, min_step + 1)
        print(f"Created looped animation data with {min_step +1 } frames")
    except Exception as e:
        print(f"Error in loop_csv(): {e }")
