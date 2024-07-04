import os

def rename_files(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Sort files to ensure consistent naming
    files.sort()

    # Loop through files and rename them
    for i, filename in enumerate(files):
        # Generate new file name
        new_name = f"{1+ i}.png"  # Change the extension as needed

        # Full old and new file paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        # Rename file
        os.rename(old_path, new_path)
        print(f"{new_path},healthy")

# Set the directory path
directory_path = "/home/mgherghinescu/projects/TenserflowModelTraining/src/model/gann/generated_images"

# Call the function
rename_files(directory_path)