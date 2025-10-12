import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageGrab
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
import math
import os
import datetime
import time

# --- Adjustable variables ---
square_size_in = 24  # Robot size (inches)
current_angle = 0  # Orientation
field_size_in = 144  # 12 ft = 144 in

# --- Data storage ---
positions = []  # [{x, y, name, note}]
square_items = []
connection_lines = []

# --- GUI Setup ---
root = tk.Tk()
root.title("Robot Game Plan GUI")

# --- Load field image ---

#TODO: to change the path of the image
image_path = "/Users/mishmash/Desktop/coading/Robot_stratagy/Field_picture/Juice-DECODE-Light.png"
if not os.path.exists(image_path):
    messagebox.showwarning("File not found",
                           f"Could not find field image:\n{image_path}\n\nPlease select manually.")
    image_path = filedialog.askopenfilename(
        title="Select Field Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )

if not image_path:
    root.destroy()
    exit()

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

# --- Canvas (left) ---
canvas_frame = tk.Frame(main_frame)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas = tk.Canvas(canvas_frame, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)

# --- Sidebar (right) ---
sidebar = tk.Frame(main_frame, width=300, bg="#000000")
sidebar.pack(side=tk.RIGHT, fill=tk.Y)
sidebar.pack_propagate(False)

tk.Label(sidebar, text="Position Editor", font=("Arial", 14, "bold"), fg="white", bg="#000000").pack(pady=10)

# Treeview
tree = ttk.Treeview(sidebar, columns=("Name", "Notes"), show="headings", height=10)
tree.heading("Name", text="Name")
tree.heading("Notes", text="Notes")
tree.pack(fill=tk.X, padx=5, pady=5)

# Inputs
tk.Label(sidebar, text="Name:", fg="white", bg="#000000").pack(anchor="w", padx=10)
name_entry = tk.Entry(sidebar)
name_entry.pack(fill=tk.X, padx=10, pady=2)

tk.Label(sidebar, text="Notes:", fg="white", bg="#000000").pack(anchor="w", padx=10)
note_entry = tk.Text(sidebar, height=4)
note_entry.pack(fill=tk.X, padx=10, pady=2)

# Labels
size_label = tk.Label(sidebar, text=f"Size: {square_size_in} in", fg="white", bg="#000000")
size_label.pack(pady=5)
angle_label = tk.Label(sidebar, text=f"Angle: {current_angle}°", fg="white", bg="#000000")
angle_label.pack(pady=5)

# --- Functions ---
def place_square(event):
    """Draw a robot square + orientation line"""
    global current_angle
    x, y = event.x, event.y
    size_px = square_size_in * px_per_in

    square_id = canvas.create_rectangle(
        x - size_px / 2, y - size_px / 2,
        x + size_px / 2, y + size_px / 2,
        outline="Black", width=5
    )

    angle_rad = math.radians(current_angle)
    line_length = size_px / 2
    x_end = x + line_length * math.sin(angle_rad)
    y_end = y - line_length * math.cos(angle_rad)
    orient_id = canvas.create_line(x, y, x_end, y_end, fill="Black", width=2)

    square_items.append((square_id, orient_id))
    positions.append({"x": x, "y": y, "name": f"Pos {len(positions) + 1}", "note": ""})

    if len(positions) > 1:
        prev = positions[-2]
        line_id = canvas.create_line(prev["x"], prev["y"], x, y, fill="blue", width=2)
        connection_lines.append(line_id)

    refresh_tree()


def refresh_tree():
    tree.delete(*tree.get_children())
    for i, pos in enumerate(positions):
        tree.insert("", "end", iid=i, values=(pos["name"], pos["note"]))


def select_position(event):
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
    selected = tree.selection()
    if not selected:
        messagebox.showwarning("No selection", "Select a position to edit.")
        return
    idx = int(selected[0])
    positions[idx]["name"] = name_entry.get()
    positions[idx]["note"] = note_entry.get("1.0", tk.END).strip()
    refresh_tree()


def adjust_size(event):
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
        steps = int(dist / 5) if dist > 0 else 1
        for step in range(steps + 1):
            x = x1 + dx * (step / steps)
            y = y1 + dy * (step / steps)
            canvas.coords(robot, x - size_px / 2, y - size_px / 2, x + size_px / 2, y + size_px / 2)
            root.update()
            time.sleep(0.1)
    canvas.delete(robot)


def export_all():
    """Export field image + notes as ONE PDF, allowing user to choose save location."""
    pdf_filename = filedialog.asksaveasfilename(
        title="Save PDF As",
        initialfile=f"robot_plan_{datetime.datetime.now().strftime('%Y-%m-%d')}.pdf",
        defaultextension=".pdf",
        filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")]
    )

    if not pdf_filename:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_img_path = f"_temp_canvas_{timestamp}.png"

    try:
        # 1. Force the window to the front to ensure it's visible for the screenshot.
        root.attributes('-topmost', True)
        root.update()
        time.sleep(1.2)  # Increased delay for rendering

        # 2. Grab the canvas content
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()
        ImageGrab.grab(bbox=(x, y, x1, y1)).save(temp_img_path)

        # --- Create PDF ---
        c = pdf_canvas.Canvas(pdf_filename, pagesize=letter)
        width, height = letter

        # Page 1: Field image
        img = Image.open(temp_img_path)
        img_width, img_height = img.size
        aspect = img_height / img_width
        display_width = width - 100
        display_height = display_width * aspect
        c.drawImage(temp_img_path, 50, height - display_height - 50, width=display_width, height=display_height)
        c.showPage()

        # Page 2: Notes
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Robot Position Notes")
        c.setFont("Helvetica", 12)
        y_cursor = height - 90
        for i, p in enumerate(positions):
            if y_cursor < 70:
                c.showPage()
                y_cursor = height - 50
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, y_cursor, "Robot Position Notes (Continued)")
                y_cursor -= 40
                c.setFont("Helvetica", 12)

            text = f"{i + 1}. {p['name']}  (x={int(p['x'])}, y={int(p['y'])})"
            c.drawString(50, y_cursor, text)
            y_cursor -= 20

            if p["note"]:
                c.setFont("Helvetica-Oblique", 11)
                note_lines = p['note'].split('\n')
                for line in note_lines:
                    c.drawString(70, y_cursor, f"Note: {line}")
                    y_cursor -= 15
                c.setFont("Helvetica", 12)
            y_cursor -= 10

        c.save()
        messagebox.showinfo("Export Complete", f"PDF saved successfully to:\n{pdf_filename}")

    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to create the PDF file.\nError: {e}")
    finally:
        # 3. ALWAYS reset the window's topmost state and clean up files.
        root.attributes('-topmost', False)
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)


# --- Buttons ---
btn_frame = tk.Frame(sidebar, bg="#000000")
btn_frame.pack(pady=0)
tk.Button(btn_frame, text="💾 Save Changes", command=save_changes).pack(side=tk.LEFT, padx=0)
tk.Button(btn_frame, text="▶ Animate Path", command=animate_path).pack(side=tk.LEFT, padx=0)
tk.Button(sidebar, text="📤 Export All (PDF)", command=export_all, bg="#000000", fg="Black").pack(fill=tk.X, padx=0,
                                                                                                 pady=0)

# --- Bindings ---
canvas.bind("<Button-1>", place_square)
root.bind("<Key>", adjust_size)
root.bind("<Delete>", delete_last_square)
tree.bind("<<TreeviewSelect>>", select_position)

# --- Start GUI ---
root.mainloop()