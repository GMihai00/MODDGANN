import json
from PIL import Image, ImageTk
import os
import tkinter as tk
import csv
import argparse
import numpy as np

#python -u "e:\Projects\TenserflowModelTraining\data_filter_helper.py" --input ./bucal_cavity_diseases_dataset/train/1/_annotations.coco.json --output E:\Projects\TenserflowModelTraining\data.csv
DISEASES_TYPES = ["OK", "pharyngitis", "tonsillitis", "mononucleosis", "healthy", "quit", "remove"]

old_image_to_disease_data = [

]

image_to_disease_data = [
    
]

validated_images = set()

def convert_path(path):
    if os.name == 'posix':
        return path.replace('\\', '/')
    else:
        return path
        
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

image_per_disease = {
    "pharyngitis" : 0,
    "tonsillitis": 0,
    "mononucleosis": 0,
    "healthy" : 0
}

def convert_image_to_bytes(image_path, expected_photo_height = 160, expected_photo_width = 160, rgb = True):
    try:
    
        image = Image.open(image_path)
        
        if not rgb:
            image = image.convert("L")
        
        width, height = image.size
        
        if height != expected_photo_height or width != expected_photo_width:
            #resize to fit the model size

            image = image.resize((expected_photo_width, expected_photo_height), Image.Resampling.LANCZOS)
        
        # normalize data
        image_array = np.array(image)
        
        image.close()
        

        return image_array.reshape(expected_photo_width, expected_photo_height, 3 if rgb else 1)
        
    except Exception as e:
        print(f"Image {image_path} not found, {repr(e)}")
        return []
        
def load_classified_images(csv_file_path):
    try:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            # skip column definition row
            next(csv_reader) 
            for row in csv_reader:
                if len(row) == 2:
                # Assuming the first column contains the values you want to insert into the set
                    image_path, disease = row
                    image_bytes = convert_image_to_bytes(convert_path(image_path))
                        
                    if len(image_bytes) != 0 and (disease in image_per_disease.keys()):
                        
                        old_image_to_disease_data.append([convert_path(image_path), disease])
                        image_per_disease[disease]+=1

    except Exception as err:
        print(err)
        exit(5)
    

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
    
def display_image_and_wait_for_choice(file_path, initial_disease, output_file):
    # Check if the image file exists
    
    if initial_disease != "tonsillitis" and initial_disease != "mononucleosis":
        return
    
    if os.path.exists(file_path):

        file_path = os.path.abspath(file_path)
        
        if file_path in validated_images:
            print(f"File {file_path} already validated")
            return
        
        print(file_path)
        img = Image.open(file_path)
        img = img.resize((500, 500))
        
        root = tk.Tk()
        root.title("Image Viewer")

        img_tk = ImageTk.PhotoImage(img)
        
        center_window(root, 500, 500)
    
        label = tk.Label(root, image=img_tk)
        label.pack()
        
        text_label = tk.Label(root, text=f"Classified before as: {initial_disease}",  font=("Helvetica", 20))
        text_label.pack()
        
        def on_button_click(option, file_path, initial_disease):
            print(f"Option {option} selected")
            
            if option == "quit":
                save_data_to_csv(output_file)
                exit(0)

            if option != "remove":
                if option != "OK":
                    image_to_disease_data.append([file_path, option])
                else:
                    image_to_disease_data.append([file_path, initial_disease])
            else:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting '{file_path}': {e}")

            root.destroy()  # Close the tkinter window when an option is selected
    
        button_width = 40  # Adjust the button width
        button_height = 1  # Adjust the button height

        for option in DISEASES_TYPES:
            button = tk.Button(root, 
                text=f"{option}", 
                width=button_width,
                height=button_height,
                command=lambda option=option, file_path=file_path, initial_disease=initial_disease: on_button_click(option, file_path, initial_disease))
            button.pack()
    
        # Start the tkinter main loop to display the image and buttons
        root.mainloop()
    else:
        print(f"Image file not found {file_path}.")


def summarize_dataset():
    total = 0
    for key, value in image_per_disease.items():
        print(f"Samples {key}: {value} images")
        total += value
        
    print(f"Total: {total} images")

def display_validated_data(output_file):
    for key, value in old_image_to_disease_data:
        # image_to_disease_data.append([key, value]) to just remove missing files 
        display_image_and_wait_for_choice(key, value, output_file);


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

    existed_before = os.path.exists(csv_file_path)
    
    try:
        with open(csv_file_path, mode="a", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
        
            if not existed_before:
                csv_writer.writerow(["Path", "Disease"])
                
            csv_writer.writerows(image_to_disease_data)
        
        remove_duplicates(csv_file_path)

    except Exception as e:
        print(f"Failed to save data to file {csv_file_path}, err: {e}")

def remove_invalid_entries(output_file):
    pass

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="Csv with validated data")
    parser.add_argument("--output", type=str, help="Csv output file path", default="data_reevaluated.csv")
    parser.add_argument("--skip_reevaluation", type=bool, help="If nothing apart from dataset details should be done", default=False)
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    reevaluate = not args.skip_reevaluation
    if input_file == None:
        print("No input file provided, quitting")
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
    
    output_file  = os.path.abspath(output_file)
        
    load_classified_images(input_file)
    
    summarize_dataset()
    
    if reevaluate:
        load_already_validated_images(output_file)
        
        display_validated_data(output_file)
        
        save_data_to_csv(output_file)
    
    print(f"Current directory: \"{os.getcwd()}\"")
    
if __name__ == "__main__":
    main()