import os
import shutil

def sort_png_by_size(folder_path):
    # Create destination folders
    noise_folder = os.path.join(folder_path, "probably_noise")
    good_folder = os.path.join(folder_path, "good_data")
    os.makedirs(noise_folder, exist_ok=True)
    os.makedirs(good_folder, exist_ok=True)

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            size_kb = os.path.getsize(file_path) / 1024  # convert to KB

            if size_kb < 1000:
                shutil.move(file_path, os.path.join(noise_folder, filename))
            else:
                shutil.move(file_path, os.path.join(good_folder, filename))

    print("Sorting complete.")

# Example usage
if __name__ == "__main__":
    
    sort_png_by_size('C:/Users/BCI-Lab/Downloads/teamA_dataset/_out_dataset')

