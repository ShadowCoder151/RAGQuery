import os

project_name = "RAGQuery"

structure = {
    "data": [],
    "src": [
        "embedding.py",
        "vector_store.py",
        "retriever.py",
        "rag_pipeline.py"
    ],
    "": [  # Root files
        "app.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "LICENSE"
    ]
}

def create_project(base_path, layout):
    for folder, files in layout.items():
        dir_path = os.path.join(base_path, folder)
        if folder:  # Skip "" for root files
            os.makedirs(dir_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'w') as f:
                pass  # Empty file placeholder

def main():
    if not os.path.exists(project_name):
        os.mkdir(project_name)
    create_project(project_name, structure)
    print(f"✅ Project structure for '{project_name}' created successfully.")

if __name__ == "__main__":
    main()
