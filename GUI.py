import tkinter as tk
from tkinter import messagebox
import numpy as np
from tensorflow.keras.models import load_model
from Prepare import *

my_model = load_model('model.keras')
pipeline = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

DEP_MAP1 = {0: "Підмет", 1: "Присудок", 2: "Обставина", 3: "Означення", 4: "Додаток", 5: "Прийменник", 6: "Пунктуація", 7: "Невідомо"}

def draw_results(words, classes):
    canvas.delete("all")
    x, y = 20, 30
    positions = []

    for word in words:
        text_id = canvas.create_text(x, y, text=word, anchor="nw", font=("Arial", 16))
        bbox = canvas.bbox(text_id)
        positions.append((bbox[0], bbox[2]))
        x = bbox[2] + 15

    for i, (word, cls) in enumerate(zip(words, classes)):
        start_x, end_x = positions[i]
        line_y = y + 25
        style = DEP_MAP1.get(cls)

        if style == "Прийменник" and i + 1 < len(classes):
            next_cls = classes[i + 1]
            if next_cls in (2, 4):
                style = DEP_MAP1[next_cls]
            else:
                continue

        if style == "Підмет":
            canvas.create_line(start_x, line_y, end_x, line_y)
        elif style == "Присудок":
            canvas.create_line(start_x, line_y, end_x, line_y)
            canvas.create_line(start_x, line_y + 3, end_x, line_y + 3)
        elif style == "Додаток":
            for j in range(start_x, end_x, 12):
                canvas.create_line(j, line_y, j + 6, line_y)
        elif style == "Обставина":
            radius = 2
            step = 14
            is_dash = True
            j = start_x
            while j < end_x:
                if is_dash:
                    canvas.create_line(j, line_y, j + 6, line_y)
                    j += 6
                else:
                    dot_x = j + radius
                    canvas.create_oval(dot_x - radius, line_y - radius, dot_x + radius, line_y + radius,
                                       fill="black", outline="black")
                    j += 2 * radius
                j += step - (6 if is_dash else 2 * radius)
                is_dash = not is_dash
        elif style == "Означення":
            for j in range(start_x, end_x, 8):
                canvas.create_arc(j, line_y - 3, j + 8, line_y + 3, start=0, extent=180, style=tk.ARC)

def on_analyze():
    text = user_entry.get().strip()

    if not text:
        messagebox.showwarning("Помилка", "Будь ласка, введіть речення.")
        return

    if any(char.isdigit() for char in text):
        messagebox.showwarning("Помилка", "Речення не повинно містити цифри.")
        return

    user_entry.delete(0, tk.END)

    tokens, vectors = analyze_text(text)
    new_data = np.array(vectors)
    predictions = my_model.predict(new_data)
    words = [fields[1] for fields in tokens]
    classes = [np.argmax(pred) for pred in predictions]
    draw_results(words, classes)

root = tk.Tk()
root.title("Розбір речення")
root.geometry("850x350")
root.configure(bg="#f2f2f2")

user_entry = tk.Entry(root, width=75, font=("Segoe UI", 14), bd=2, relief="groove", highlightthickness=1)
user_entry.pack(pady=20, ipady=6)

style_btn = {"font": ("Segoe UI", 12, "bold"),
             "bg": "#4285F4",
             "fg": "white",
             "activebackground": "#3367D6",
             "activeforeground": "white",
             "bd": 0,
             "padx": 20,
             "pady": 8,
             "cursor": "hand2"}

analyze_btn = tk.Button(root, text="Аналізувати", command=on_analyze, **style_btn)
analyze_btn.pack()

canvas_frame = tk.Frame(root, bg="#e0e0e0", bd=2, relief="groove")
canvas_frame.pack(pady=20, padx=20, fill="both", expand=True)

canvas = tk.Canvas(canvas_frame, bg="white", height=100, highlightthickness=0)
canvas.pack(fill="both", expand=True)

root.mainloop()