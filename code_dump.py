# smart_project_dump.py
import os

root = "C:\\Users\\lenovo\\Documents\\Thesis\\thesis\\GCL\\Thesis_2\\GCL_V1"  # Change this to your project root if needed
output_file = "project_dump.txt"

# Directories and files to completely skip
EXCLUDE_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    ".next",          # Next.js build/cache
    "build",
    "dist",
    "out",
    "coverage",
    ".vscode",
    ".idea",
    "venv",
    ".venv",
    "env",
    ".env",
    "public",         # usually static assets, optional
    "static",
    "data",
    "lightning_logs",
    "results",
    # Django/static files
}

# File patterns to skip (even if in allowed folders)
EXCLUDE_FILES = {
    ".log",
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    ".DS_Store",
    "Thumbs.db",
}

# Only include these extensions (add more if needed)
INCLUDE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".html", ".css", ".scss", ".yaml", ".yml", ".env.example"}

def should_include_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in INCLUDE_EXTENSIONS

def should_exclude_path(path):
    parts = path.replace("\\", "/").split("/")  # Normalize for Windows
    return any(part in EXCLUDE_DIRS for part in parts) or any(part.startswith(".git") for part in parts)

with open(output_file, "w", encoding="utf-8") as out:
    out.write(f"# Project Code Dump: {root}\n")
    out.write(f"# Generated on: {os.path.abspath(root)}\n\n")

    for current_folder, dirs, files in os.walk(root):
        # Skip excluded directories in-place (modifies dirs to prevent walking into them)
        dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(current_folder, d))]

        if should_exclude_path(current_folder):
            continue

        for file in files:
            if file in EXCLUDE_FILES or file.startswith("."):
                continue
            if not should_include_file(file):
                continue

            filepath = os.path.join(current_folder, file)
            rel_path = os.path.relpath(filepath, root)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                content = f"[ERROR READING FILE: {e}]"

            out.write(f"\n\n===== FILE: {rel_path} =====\n\n")
            out.write(content)

print(f"Code dump successfully written to {output_file}")
print("Excluded: node_modules, .git, build folders, lockfiles, env files, etc.")