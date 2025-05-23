#!/usr/bin/env python3

import subprocess
import time

bag_files = [
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_12_36",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_18_38",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_33_59",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_39_08",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-01_50_26",
    "src/smarc_modelling/piml/data/prebags/rosbag2_1970_01_01-02_09_13"
]

def play_bag(bag_path):
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
    for bag in bag_files:
        play_bag(bag)
        time.sleep(2) # Small delay between each bag

if __name__ == "__main__":
    main()