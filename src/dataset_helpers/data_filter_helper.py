import json
from PIL import Image, ImageTk
import os
import tkinter as tk
import csv
import argparse

#python -u "e:\Projects\TenserflowModelTraining\data_filter_helper.py" --input ./bucal_cavity_diseases_dataset/train/1/_annotations.coco.json --output E:\Projects\TenserflowModelTraining\data.csv
DISEASES_TYPES = ["pharyngitis", "tonsillitis", "mononucleosis", "healthy", "inconcludent", "quit"]

image_to_disease_data = [

]

validated_images = set()


def load_already_validated_images(csv_file_path):
    # Open the CSV file and read the values from the first column
    try:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            cnt = 0
            for row in csv_reader:
                # Assuming the first column contains the values you want to insert into the set
                cnt+=1
                validated_images.add(row[0])
                
            if cnt == 0:
                image_to_disease_data.append(["Path", "Disease"])
    except FileNotFoundError:
        print(f"The file {csv_file_path} does not exist, creating it")
        image_to_disease_data.append(["Path", "Disease"])
        with open(csv_file_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)


def center_window(root, photo_height, photo_width):
    # Calculate the screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    window_width = photo_width + 400
    window_height = photo_height + 550
    # Calculate the window position for centering
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    
    # Set the window geometry
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    
def display_image_and_wait_for_choice(file_path, photo_height, photo_width, output_file):
    # Check if the image file exists
    if os.path.exists(file_path):

        file_path = os.path.abspath(file_path)
        # Open the image using PIL
        if file_path in validated_images:
            print(f"File {file_path} already validated")
            return
        
        img = Image.open(file_path)
        img = img.resize((500, 500))
        
        root = tk.Tk()
        root.title("Image Viewer")

        img_tk = ImageTk.PhotoImage(img)
        
        center_window(root, 500, 500)
    
        label = tk.Label(root, image=img_tk)
        label.pack()
        
        def on_button_click(option, file_path):
            print(f"Option {option} selected")
            
            if option == "quit":
                save_data_to_csv(output_file)
                exit(0)

            if option != "inconcludent":
                image_to_disease_data.append([file_path, option])
            else:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting '{file_path}': {e}")

            root.destroy()  # Close the tkinter window when an option is selected
    
        button_width = 40  # Adjust the button width
        button_height = 1  # Adjust the button height

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

def parse_coco_json(coco_json_path, output_file):
    with open(coco_json_path, 'r') as file:
        data = json.load(file)
        
        for elem in data["images"]:
            display_image_and_wait_for_choice(elem["file_name"], elem["height"], elem["width"], output_file)


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

    print(f"Saving file to: {csv_file_path}")

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
    
    output_file  = os.path.abspath(output_file)
    
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
    
    load_already_validated_images(output_file)
    
    parse_coco_json(input_file, output_file)
    
    save_data_to_csv(output_file)

if __name__ == "__main__":
    main()