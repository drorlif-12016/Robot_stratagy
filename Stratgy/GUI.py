import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageGrab
import math
import os
import datetime
import time

# --- Adjustable variables ---
square_size_in = 24  # Robot size (inches)
current_angle = 0  # Orientation
field_size_in = 144  # 12 ft = 144 in

# --- Data storage ---
positions = []  # [(x, y, name, note)]
square_items = []
connection_lines = []

# --- GUI Setup ---
root = tk.Tk()
root.title("Robot Game Plan GUI")

# --- Load field image ---
image_path = "Enter root for image here"
if not os.path.exists(image_path):
    messagebox.showwarning("File not found",
                           f"Could not find field image:\n{image_path}\n\nPlease select manually.")
    image_path = filedialog.askopenfilename(
        title="Select Field Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

img = Image.open(image_path)
window_size = 600
scale_factor = min(window_size / img.width, window_size / img.height)
img = img.resize((int(img.width * scale_factor), int(img.height * scale_factor)))
tk_img = ImageTk.PhotoImage(img)

# --- Conversion ---
px_per_in = img.width / field_size_in

# --- Main layout ---
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# --- Canvas on left ---
canvas_frame = tk.Frame(main_frame)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas = tk.Canvas(canvas_frame, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)

# --- Sidebar on right ---
sidebar = tk.Frame(main_frame, width=300, bg="#f0f0f0")
sidebar.pack(side=tk.RIGHT, fill=tk.Y)
sidebar.pack_propagate(False)

tk.Label(sidebar, text="Position Editor", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

# Treeview for list of positions
tree = ttk.Treeview(sidebar, columns=("Name", "Notes"), show="headings", height=10)
tree.heading("Name", text="Name")
tree.heading("Notes", text="Notes")
tree.pack(fill=tk.X, padx=5, pady=5)

# Inputs
tk.Label(sidebar, text="Name:", bg="#f0f0f0").pack(anchor="w", padx=10)
name_entry = tk.Entry(sidebar)
name_entry.pack(fill=tk.X, padx=10, pady=2)

tk.Label(sidebar, text="Notes:", bg="#f0f0f0").pack(anchor="w", padx=10)
note_entry = tk.Text(sidebar, height=4)
note_entry.pack(fill=tk.X, padx=10, pady=2)

# Status labels
size_label = tk.Label(sidebar, text=f"Size: {square_size_in} in", bg="#f0f0f0")
size_label.pack(pady=5)
angle_label = tk.Label(sidebar, text=f"Angle: {current_angle}°", bg="#f0f0f0")
angle_label.pack(pady=5)


# --- Functions ---
def place_square(event):
    """Place a robot position with square and line"""
    global current_angle
    x, y = event.x, event.y
    size_px = square_size_in * px_per_in

    square_id = canvas.create_rectangle(
        x - size_px / 2, y - size_px / 2,
        x + size_px / 2, y + size_px / 2,
        outline="red", width=2
    )

    angle_rad = math.radians(current_angle)
    line_length = size_px / 2
    x_end = x + line_length * math.sin(angle_rad)
    y_end = y - line_length * math.cos(angle_rad)
    orient_id = canvas.create_line(x, y, x_end, y_end, fill="green", width=2)

    square_items.append((square_id, orient_id))
    positions.append({"x": x, "y": y, "name": f"Pos {len(positions)+1}", "note": ""})

    # Add connection line
    if len(positions) > 1:
        prev = positions[-2]
        line_id = canvas.create_line(prev["x"], prev["y"], x, y, fill="blue", width=2)
        connection_lines.append(line_id)

    # Update sidebar
    refresh_tree()


def refresh_tree():
    """Refresh sidebar list"""
    for i in tree.get_children():
        tree.delete(i)
    for i, pos in enumerate(positions):
        tree.insert("", "end", iid=i, values=(pos["name"], pos["note"]))


def select_position(event):
    """When clicking a row in sidebar"""
    selected = tree.selection()
    if not selected:
        return
    idx = int(selected[0])
    pos = positions[idx]
    name_entry.delete(0, tk.END)
    name_entry.insert(0, pos["name"])
    note_entry.delete("1.0", tk.END)
    note_entry.insert("1.0", pos["note"])


def save_changes():
    """Update position info"""
    selected = tree.selection()
    if not selected:
        messagebox.showwarning("No selection", "Please select a position to edit.")
        return
    idx = int(selected[0])
    positions[idx]["name"] = name_entry.get()
    positions[idx]["note"] = note_entry.get("1.0", tk.END).strip()
    refresh_tree()


def adjust_size(event):
    """W/S for size, Q/E for angle"""
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
    """Delete last position"""
    if not square_items:
        return
    square_id, orient_id = square_items.pop()
    canvas.delete(square_id)
    canvas.delete(orient_id)
    if connection_lines:
        canvas.delete(connection_lines.pop())
    positions.pop()
    refresh_tree()


def animate_path():
    """Animate robot movement"""
    if len(positions) < 2:
        messagebox.showwarning("No Path", "Place at least two positions to animate.")
        return

    size_px = square_size_in * px_per_in
    robot = canvas.create_rectangle(0, 0, size_px, size_px, outline="yellow", width=3)

    for i in range(len(positions) - 1):
        x1, y1 = positions[i]["x"], positions[i]["y"]
        x2, y2 = positions[i + 1]["x"], positions[i + 1]["y"]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        steps = int(dist / 5)
        for step in range(steps + 1):
            x = x1 + dx * (step / steps)
            y = y1 + dy * (step / steps)
            canvas.coords(robot, x - size_px / 2, y - size_px / 2, x + size_px / 2, y + size_px / 2)
            root.update()
            time.sleep(0.01)
    canvas.delete(robot)


def export_all():
    """Export field + text summary"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_filename = f"field_export_{timestamp}.png"
    text_filename = f"field_notes_{timestamp}.txt"

    # Save canvas as image
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    ImageGrab.grab(bbox=(x, y, x1, y1)).save(img_filename)

    # Save notes
    with open(text_filename, "w", encoding="utf-8") as f:
        f.write("Robot Position Notes\n")
        f.write("====================\n\n")
        for i, p in enumerate(positions):
            f.write(f"{i+1}. {p['name']}\n")
            f.write(f"   Note: {p['note']}\n")
            f.write(f"   Coordinates: ({int(p['x'])}, {int(p['y'])})\n\n")

    messagebox.showinfo("Export Complete",
                        f"Saved:\n• {img_filename}\n• {text_filename}")


# --- Buttons ---
btn_frame = tk.Frame(sidebar, bg="#f0f0f0")
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="💾 Save Changes", command=save_changes).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="▶ Animate Path", command=animate_path).pack(side=tk.LEFT, padx=5)
tk.Button(sidebar, text="📤 Export All", command=export_all, bg="#d0ffd0").pack(fill=tk.X, padx=10, pady=10)

# --- Bindings ---
canvas.bind("<Button-1>", place_square)
root.bind("<Key>", adjust_size)
root.bind("<Delete>", delete_last_square)
tree.bind("<<TreeviewSelect>>", select_position)

# --- Start GUI ---
root.mainloop()
