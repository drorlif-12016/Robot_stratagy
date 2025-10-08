import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import math
import os

# --- Adjustable variables (in INCHES) ---
square_size_in = 24   # Robot size in inches
square_centers = []   # List to store square centers
square_items = []     # Store square + orientation line canvas IDs
connection_lines = [] # Store connecting line canvas IDs
current_angle = 0     # Angle for new squares (degrees)

# --- Real field dimensions (12 ft = 144 inches) ---
field_size_in = 144

# --- Main Window ---
root = tk.Tk()
root.title("Game Plan GUI")

# --- Load background field image ---
image_path = "Enter root for image here"

if not os.path.exists(image_path):
    messagebox.showwarning("File not found",
                           f"Could not find field image:\n{image_path}\n\nPlease select manually.")
    image_path = filedialog.askopenfilename(
        title="Select Field Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

img = Image.open(image_path)

# Fit image into window (maintain aspect ratio)
window_size = 600
scale_factor = min(window_size / img.width, window_size / img.height)
new_width = int(img.width * scale_factor)
new_height = int(img.height * scale_factor)
img = img.resize((new_width, new_height))
tk_img = ImageTk.PhotoImage(img)

# --- Conversion: inches → pixels ---
px_per_in_x = img.width / field_size_in
px_per_in_y = img.height / field_size_in
px_per_in = min(px_per_in_x, px_per_in_y)

# --- Canvas ---
canvas = tk.Canvas(root, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)

# --- Functions ---
def place_square(event):
    """Draw a robot square where the user clicks with orientation line"""
    x, y = event.x, event.y

    # Convert size from inches to pixels
    size_px = square_size_in * px_per_in

    # Draw square
    square_id = canvas.create_rectangle(
        x - size_px // 2, y - size_px // 2,
        x + size_px // 2, y + size_px // 2,
        outline="red", fill="", width=2
    )

    # Orientation line
    angle_rad = math.radians(current_angle)
    line_length = size_px / 2
    x_end = x + line_length * math.sin(angle_rad)
    y_end = y - line_length * math.cos(angle_rad)
    orient_line_id = canvas.create_line(x, y, x_end, y_end, fill="green", width=2)

    # Store square and orientation line IDs
    square_items.append((square_id, orient_line_id))
    square_centers.append((x, y))

    # Draw connection line if needed
    if len(square_centers) > 1:
        x1, y1 = square_centers[-2]
        x2, y2 = square_centers[-1]
        line_id = canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)
        connection_lines.append(line_id)


def adjust_size(event):
    """Adjust robot size or angle"""
    global square_size_in, current_angle

    if event.keysym == "w":
        square_size_in += 1
    elif event.keysym == "s":
        square_size_in = max(1, square_size_in - 1)
    elif event.keysym == "q":
        current_angle = (current_angle - 10) % 360
    elif event.keysym == "e":
        current_angle = (current_angle + 10) % 360

    size_label.config(text=f"Size: {square_size_in} in")
    angle_label.config(text=f"Angle: {current_angle}°")


def delete_last_square(event=None):
    """Delete the most recently drawn square, orientation line, and connection line"""
    if square_items:
        square_id, orient_line_id = square_items.pop()
        canvas.delete(square_id)
        canvas.delete(orient_line_id)
        square_centers.pop()

        if connection_lines:
            last_line_id = connection_lines.pop()
            canvas.delete(last_line_id)


# --- Labels ---
size_label = tk.Label(root, text=f"Size: {square_size_in} in")
size_label.pack()
angle_label = tk.Label(root, text=f"Angle: {current_angle}°")
angle_label.pack()

# --- Bindings ---
canvas.bind("<Button-1>", place_square)
root.bind("<Key>", adjust_size)
root.bind("<Delete>", delete_last_square)

# --- Start GUI ---
root.mainloop()
