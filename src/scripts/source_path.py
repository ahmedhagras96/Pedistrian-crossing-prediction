import sys
import os

# Add the src directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "..")
sys.path.insert(0, src_path)

# Optional: Print sys.path to verify
# print(f"sys.path: {sys.path}")
