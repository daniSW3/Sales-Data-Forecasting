import os
from pathlib import Path

# Define the folder structure
structure = {
    ".vscode": ["settings.json"],
    ".github": {
        "workflows": ["unittests.yml"]
    },
    "src": [],
    "notebooks": ["__init__.py", "README.md"],
    "tests": ["__init__.py"],
    "scripts": ["__init__.py", "README.md"],
}

# Define root-level files
root_files = [".gitignore", "requirements.txt", "README.md"]

def create_structure(base_path, struct):
    for name, content in struct.items():
        folder_path = Path(base_path) / name
        folder_path.mkdir(parents=True, exist_ok=True)

        if isinstance(content, dict):  # nested folders
            create_structure(folder_path, content)
        elif isinstance(content, list):  # files inside this folder
            for file_name in content:
                file_path = folder_path / file_name
                file_path.touch(exist_ok=True)

if __name__ == "__main__":
    base_dir = Path(".")  # current directory

    # Create folder structure
    create_structure(base_dir, structure)

    # Create root-level files
    for file in root_files:
        Path(base_dir / file).touch(exist_ok=True)

    print("✅ Project structure created successfully!")
