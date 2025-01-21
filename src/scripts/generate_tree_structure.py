import os

def generate_tree(directory, indent=''):
    tree = []
    for item in os.listdir(directory):
        # Skip __pycache__ directories
        if item == "__pycache__":
            continue
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            tree.append(f"{indent}├── {item}/")
            tree.append(generate_tree(path, indent + "│   "))
        else:
            tree.append(f"{indent}├── {item}")
    return "\n".join(tree)

# Save the tree structure to a file with UTF-8 encoding
with open("../tree.txt", "w", encoding="utf-8") as f:
    f.write(generate_tree("../"))