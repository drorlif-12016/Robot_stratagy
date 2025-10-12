# Decode 2025–2026 — Robot Strategy Toolkit

This is a dedicated **robot strategy planning** toolkit for the 2025–2026 *Decode* game.  
The purpose of this toolkit is to help teams develop and optimize gameplay strategies for this season’s challenge.

---

## 📂 Files Included

### 1. `Points_calculator_unsorted.py`
A points calculator that **does not factor in pattern points** — it only uses:
- The robot’s **cycle time**  
- The **number of times** the gate will open  

**Output:**
- Recommended number of times that the gate should be opened In every match period 
- Points breakdown between **classified artifacts** and **overflow artifacts**  

> **Note:**  
> The algorithm favors gate openings during the **TeleOp period**.  
> User discretion is advised when adjusting strategy timing.

---

### 2. `Points_calculator_sorted.py`
This version **includes pattern points** and requires additional inputs:
- Number of **sorted cycles** performed  
- Time required for each sorted cycle  

💡 To get the most accurate results, the **sorted cycle time** should be *longer* than the unsorted cycle time.

> **Note Agin:**  
> The algorithm favors gate openings during the **TeleOp period**.  
> User discretion is advised when adjusting strategy timing.
---

### 3. Scouting App
A companion app designed for data collection and analysis during matches.  
It helps teams Decide their alliance partners based on OPR performance and 
strategize for the last parts of the tournament based on real-time data. 
form the FIRST API 
the Documentation could be found in this link: 
#(add a link to the documentation)#

---
## the GUI 
this part of the strategy toolkit helps you visualize your robots actions 
with the Function to add notes to any position on the field and also name said 
position and then easily export it to A .PDF file format 

## 🕹️ GUI Controls

| Action               | Key(s)       | Description                        |
|----------------------|--------------|------------------------------------|
| Rotate robot         | `Q` / `E`    | Adjust the robot’s orientation     |
| Resize th robot      | `W` / `S`    | Adds size to the robot / decrease  |
| Add new position     | Mouse Click  | Adds a robot position on the field |
| Delete last position | `Delete` key | Removes the most recent position   |

---

## ⚙️ Robot Scaling

The robot field display is **1:1 scale**.  
- A **24-inch robot** equals a **24-inch tile** on the field.

---

### 🧩 Summary

This toolkit provides:
- A **point calculator** (sorted and unsorted versions)  
- A **visual GUI field tool** with animation and note-tracking 
- A **scouting app** for real-time match data and better alliance partner selection 

Together, they form a complete system for **robot gameplay planning** and **performance optimization** for the *Decode 2025–2026* season.
