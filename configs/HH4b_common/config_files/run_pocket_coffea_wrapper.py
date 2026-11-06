import sys
import os

def main():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Replace the python process with bash so that signals (e.g. Ctrl+C) reach
    # the bash script directly and its cleanup trap can run
    os.execvp("bash", ["bash", f"{file_dir}/run_pocket_coffea.sh"] + sys.argv[1:])
