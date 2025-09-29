import tkinter as tk
from PIL import Image, ImageTk
import math

# --- Adjustable variables (in CM) ---
square_width_cm = 20   # horizontal size in cm
square_height_cm = 20  # vertical size in cm
square_centers = []    # List to store square centers
current_angle = 0      # Angle for new squares (degrees)

# --- Real field dimensions ---
field_size_cm = 366  # 12 ft = 366 cm

# --- Main Window ---
root = tk.Tk()
root.title("Game Plan GUI")

# Load background field image
image_path = "/Users/mishmash/Desktop/coading/Robot_stratagy/Field top view .png"  # <-- replace with your image
img = Image.open(image_path)
img = img.resize((1100, 800))  # keep your field photo resolution
tk_img = ImageTk.PhotoImage(img)

# --- Conversion: cm → pixels (based on image and real field size) ---
px_per_cm_x = img.width / field_size_cm
px_per_cm_y = img.height / field_size_cm

# Use the smaller value to keep robots square
px_per_cm = min(px_per_cm_x, px_per_cm_y)


def place_square(event):
    """Draw a square where the user clicks with orientation line"""
    x, y = event.x, event.y

    # Convert size from cm to pixels
    width_px = square_width_cm * px_per_cm
    height_px = square_height_cm * px_per_cm

    # Draw square
    canvas.create_rectangle(
        x - width_px // 2, y - height_px // 2,
        x + width_px // 2, y + height_px // 2,
        outline="red", fill="", width=2
    )

    # Orientation line (from center, in direction of current_angle)
    angle_rad = math.radians(current_angle)
    line_length = height_px / 2
    x_end = x + line_length * math.sin(angle_rad)
    y_end = y - line_length * math.cos(angle_rad)
    canvas.create_line(x, y, x_end, y_end, fill="green", width=2)

    # Save center
    square_centers.append((x, y))

    # Connect previous squares
    if len(square_centers) > 1:
        x1, y1 = square_centers[-2]
        x2, y2 = square_centers[-1]
        canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)


def adjust_size(event):
    """Adjust square size (cm) or angle"""
    global square_width_cm, square_height_cm, current_angle

    if event.keysym == "w":
        square_height_cm += 1
    elif event.keysym == "s":
        square_height_cm = max(1, square_height_cm - 1)
    elif event.keysym == "d":
        square_width_cm += 1
    elif event.keysym == "a":
        square_width_cm = max(1, square_width_cm - 1)
    elif event.keysym == "q":
        current_angle = (current_angle - 10) % 360
    elif event.keysym == "e":
        current_angle = (current_angle + 10) % 360

    size_label.config(text=f"Size: {square_width_cm}cm x {square_height_cm}cm")
    angle_label.config(text=f"Angle: {current_angle}°")


# --- Canvas ---
canvas = tk.Canvas(root, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)

# --- Labels ---
size_label = tk.Label(root, text=f"Size: {square_width_cm}cm x {square_height_cm}cm")
size_label.pack()
angle_label = tk.Label(root, text=f"Angle: {current_angle}°")
angle_label.pack()

# --- Bind events ---
canvas.bind("<Button-1>", place_square)
root.bind("<Key>", adjust_size)

root.mainloop()
