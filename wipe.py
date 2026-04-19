import shutil
import os
from pathlib import Path

def clean():
    for folder in ['storage/metadata', 'storage/models', 'storage/audit']:
        p = Path(folder)
        if p.exists():
            for child in p.iterdir():
                if child.is_file() and child.name != '.gitkeep':
                    child.unlink()
    
    print("Storage safely cleared.")

if __name__ == '__main__':
    clean()
