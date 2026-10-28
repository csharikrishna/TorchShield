"""Script to create a clean, reproducible zip archive of the TorchShield codebase."""
import os
import zipfile

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "build",
    "torchshield.egg-info",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".zip",
    ".tmp",
}

def make_clean_zip(output_zip_path: str, root_dir: str = "."):
    abs_root = os.path.abspath(root_dir)
    print(f"Archiving from: {abs_root}")
    print(f"Output archive: {output_zip_path}")

    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(abs_root):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDE_EXTS:
                    continue

                full_path = os.path.join(root, file)
                # Compute relative path inside archive with top-level folder 'TorchShield/'
                rel_path = os.path.relpath(full_path, abs_root)
                arc_name = os.path.join("TorchShield", rel_path)

                zipf.write(full_path, arc_name)
                file_count += 1
                total_bytes += os.path.getsize(full_path)

    zip_size = os.path.getsize(output_zip_path)
    print(f"Archive created successfully!")
    print(f"Files included: {file_count}")
    print(f"Uncompressed size: {total_bytes:,} bytes")
    print(f"Compressed zip size: {zip_size:,} bytes")
    return output_zip_path, file_count, zip_size

if __name__ == "__main__":
    os.makedirs("archives", exist_ok=True)
    out_archive = os.path.join("archives", "torchshield_v0.7.0.zip")
    make_clean_zip(out_archive)

    # Also place a copy at parent folder for direct download convenience
    parent_archive = os.path.abspath(os.path.join("..", "torchshield_v0.7.0.zip"))
    import shutil
    shutil.copy2(out_archive, parent_archive)
    print(f"Copied archive to parent folder: {parent_archive}")
