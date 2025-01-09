import sys
import os

def add_source_path():
  # Add the src directory to sys.path
  current_dir = os.path.dirname(os.path.abspath(__file__))
  src_path = os.path.dirname(os.path.join(current_dir))
  modules_path = os.path.join(src_path, "modules")
  sys.path.insert(0, src_path)
  sys.path.insert(0, modules_path)

  # Optional: Print sys.path to verify
  print(f"sys.path: {sys.path}")


add_source_path()