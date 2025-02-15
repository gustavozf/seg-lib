import hashlib
import os

def hash_folder(folder_path):
    sha512 = hashlib.sha512()

    for root, dirs, files in sorted(os.walk(folder_path)):
        # Include directory names in the hash
        for dir_name in sorted(dirs):
            dir_path = os.path.join(root, dir_name)

            if '__pycache__' in dir_path:
                continue

            print(dir_path)
            sha512.update(dir_path.encode('utf-8'))
        
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            
            if '__pycache__' in file_path:
                continue
            
            print(file_path)
            sha512.update(file_path.encode('utf-8'))  # Include file path in hash
            
            # Read file content and update hash
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha512.update(chunk)

    return sha512.hexdigest()

# Example usage
folder_path = "path/to/folder"
folder_hash = hash_folder(folder_path)
print(f"SHA-512 hash of the folder: {folder_hash}")
