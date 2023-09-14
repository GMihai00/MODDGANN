import json
from PIL import Image, ImageTk
import os
import time
import tkinter as tk

DISEASES_TYPES = ["Laryngitis", "pharyngitis", "tonsillitis", "gastric reflux", "tonsil stones", "none"]

# can be taken from input
try:
    os.chdir("./bucal_cavity_diseases_dataset/train")
except:
    pass

coco_json_path = "_annotations.coco.json"


with open(coco_json_path, 'r') as file:
    data = json.load(file)
    
    
    for elem in data["images"]:

        # Check if the image file exists
        if os.path.exists(elem["file_name"]):
            # Open the image using PIL
            img = Image.open(elem["file_name"])
            
            # Create a tkinter root window
            root = tk.Tk()
            root.title("Image Viewer")
    
            # Convert the image to a format compatible with tkinter
            img_tk = ImageTk.PhotoImage(img)
            
            # Calculate the screen dimensions
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            
            window_width = elem["height"]
            window_height = elem["width"] + 250
            # Calculate the window position for centering
            x_position = (screen_width - window_width) // 2
            y_position = (screen_height - window_height) // 2
            
            # Set the window geometry
            root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
            # Display the image in a tkinter Label widget
            label = tk.Label(root, image=img_tk)
            label.pack()
            
            # Define a function to handle button clicks
            def on_button_click(option):
                print(f"Option {option} selected")
                root.destroy()  # Close the tkinter window when an option is selected
        
            button_width = 40  # Adjust the button width
            button_height = 2  # Adjust the button height

            print(button_height)
            for option in DISEASES_TYPES:
                button = tk.Button(root, 
                    text=f"{option}", 
                    width=button_width,
                    height=button_height,
                    command=lambda option=option: on_button_click(option))
                button.pack()
        
            # Start the tkinter main loop to display the image and buttons
            root.mainloop()
        else:
            print("Image file not found.")