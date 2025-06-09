#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import time

# List of the bags we want to play
bag_files_2025 = [
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_20_12",
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_25_16",
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_26_45",
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_27_08",
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_29_34",
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_32_03",
    "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_37_01",
]

bag_files_1970 = [
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_12_36",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_18_38",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_33_59",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_39_08",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_50_26",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-02_09_13"
]

bag_files_combine = [
    "src/smarc_modelling/piml/data/rosbags/rosbag_tank_1970",
    "src/smarc_modelling/piml/data/rosbags/rosbag_tank_2025"
]

# "src/smarc_modelling/piml/data/prebags/rosbag2_2025_05_22-15_22_01", We are saving this one for testing in the end

def play_bag(bag_path):
    """Plays the above list of bags one after another for easy combining into one set of training data"""
    print(f" Playing bag: {bag_path}")

    process = subprocess.Popen(
        ["ros2", "bag", "play", bag_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    try:
        for line in process.stdout:
            print(line, end="")
    except KeyboardInterrupt:
        process.terminate()
        print(f" Stopped playing bags!")

    process.wait()
    print(f" Finished playing {bag_path}")


def main():

    # Quick swap between the two different sets, 1970 uses mocap and 2025 uses mocap2
    selected_bags = bag_files_2025

    for bag in selected_bags:
        play_bag(bag)
        time.sleep(1) # Small delay between each bag
    


if __name__ == "__main__":
    main()