import tkinter as tk
from PIL import Image, ImageTk
import math

# --- Adjustable variables (in INCHES) ---
square_size_in = 18   # robot size (in inches, example: 18" x 18")
square_centers = []   # List to store square centers
current_angle = 0     # Angle for new squares (degrees)

# --- Real field dimensions (12 ft = 144 inches) ---
field_size_in = 144

# --- Main Window ---
root = tk.Tk()
root.title("Game Plan GUI")

# Load background field image
image_path = "Enter field pic path"  # field photo you uploaded
img = Image.open(image_path)

# Fit the image into a window (keep aspect ratio)
window_size = 800
scale_factor = min(window_size / img.width, window_size / img.height)
new_width = int(img.width * scale_factor)
new_height = int(img.height * scale_factor)
img = img.resize((new_width, new_height))
tk_img = ImageTk.PhotoImage(img)

# --- Conversion: inches → pixels (based on image and real field size) ---
px_per_in_x = img.width / field_size_in
px_per_in_y = img.height / field_size_in
px_per_in = min(px_per_in_x, px_per_in_y)  # keep square ratio


def place_square(event):
    """Draw a robot square where the user clicks with orientation line"""
    x, y = event.x, event.y

    # Convert robot size from inches to pixels
    size_px = square_size_in * px_per_in

    # Draw square robot
    canvas.create_rectangle(
        x - size_px // 2, y - size_px // 2,
        x + size_px // 2, y + size_px // 2,
        outline="red", fill="", width=2
    )

    # Orientation line (from center, in direction of current_angle)
    angle_rad = math.radians(current_angle)
    line_length = size_px / 2
    x_end = x + line_length * math.sin(angle_rad)
    y_end = y - line_length * math.cos(angle_rad)
    canvas.create_line(x, y, x_end, y_end, fill="green", width=2)

    # Save center
    square_centers.append((x, y))

    # Connect previous robots
    if len(square_centers) > 1:
        x1, y1 = square_centers[-2]
        x2, y2 = square_centers[-1]
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)


def adjust_size(event):
    """Adjust robot size (in inches) or angle"""
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


# --- Canvas ---
canvas = tk.Canvas(root, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)

# --- Labels ---
size_label = tk.Label(root, text=f"Size: {square_size_in} in")
size_label.pack()
angle_label = tk.Label(root, text=f"Angle: {current_angle}°")
angle_label.pack()

# --- Bind events ---
canvas.bind("<Button-1>", place_square)
root.bind("<Key>", adjust_size)

root.mainloop()
