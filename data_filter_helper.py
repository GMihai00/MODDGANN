import json
from PIL import Image, ImageTk
import os
import tkinter as tk
import csv
import argparse

DISEASES_TYPES = ["Laryngitis", "pharyngitis", "tonsillitis", "gastric reflux", "tonsil stones", "none"]

image_to_disease_data = [
    ["Path", "Disease"]
]

def center_window(root, photo_height, photo_width):
    # Calculate the screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    window_width = photo_width 
    window_height = photo_height + 250
    # Calculate the window position for centering
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    
    # Set the window geometry
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
def display_image_and_wait_for_choice(file_path, photo_height, photo_width):
    # Check if the image file exists
    if os.path.exists(file_path):

        file_path = os.path.abspath(file_path)
        # Open the image using PIL
        img = Image.open(file_path)
        
        root = tk.Tk()
        root.title("Image Viewer")

        img_tk = ImageTk.PhotoImage(img)
        
        center_window(root, photo_height, photo_width)
    
        label = tk.Label(root, image=img_tk)
        label.pack()
        
        def on_button_click(option, file_path):
            print(f"Option {option} selected")
            image_to_disease_data.append([file_path, option])
            root.destroy()  # Close the tkinter window when an option is selected
    
        button_width = 40  # Adjust the button width
        button_height = 2  # Adjust the button height

        print(button_height)
        for option in DISEASES_TYPES:
            button = tk.Button(root, 
                text=f"{option}", 
                width=button_width,
                height=button_height,
                command=lambda option=option, file_path=file_path: on_button_click(option, file_path))
            button.pack()
    
        # Start the tkinter main loop to display the image and buttons
        root.mainloop()
    else:
        print(f"Image file not found {file_path}.")

def parse_coco_json(coco_json_path):
    with open(coco_json_path, 'r') as file:
        data = json.load(file)
        
        for elem in data["images"]:
            display_image_and_wait_for_choice(elem["file_name"], elem["height"], elem["width"])


def remove_duplicates(input_path):
    # Create a set to store unique rows
    unique_rows = set()

    # Read data from the input CSV file and remove duplicates
    with open(input_path, mode="r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Read the header row (if present)
        
        for row in csv_reader:
            # Convert the row to a tuple to make it hashable
            row_tuple = tuple(row)
            unique_rows.add(row_tuple)

    # Write deduplicated data to the output CSV file
    with open(input_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        
        for row_tuple in unique_rows:
            csv_writer.writerow(row_tuple)

def save_data_to_csv(csv_file_path):
    try:
        with open(csv_file_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
        
            csv_writer.writerows(image_to_disease_data)
        
        remove_duplicates(csv_file_path)

    except Exception as e:
        print(f"Failed to save data to file {csv_file_path}, err: {e}")

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="Coco json input file path")
    parser.add_argument("--output", type=str, help="Csv output file path", default="data.csv")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    if input_file == None:
        print("No input file provided, quiting")
        exit(5)
    
    if os.path.exists(input_file):
        #convert to absolute path
        input_file = os.path.abspath(input_file)
        # Split the path into parent directory and file name
        parent_directory, file_name = os.path.split(input_file)
        
        try:
            os.chdir(parent_directory)
        except:
            print(f"Failed to change to target dir {parent_directory}")
            pass
    
        input_file = file_name
    else:
        print(f"Can't find {input_file}")
        exit(5)
        
    parse_coco_json(input_file)
    
    save_data_to_csv(output_file)

if __name__ == "__main__":
    main()