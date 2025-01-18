import os
import sys


def configure_main():
    # Sets the source path to ensure that the program can locate module files correctly using sys.path.
    # Appends the directory of the current file to the Python module search path to ensure modules in the same directory can be imported.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Add the src directory to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, "..")
    sys.path.insert(0, src_path)