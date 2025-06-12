import os
import pickle
import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    return tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 20)
    )


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    return tk.Text(window, height=2, width=15, font=("Arial", 32))


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path):
    """
    Recognize a face from the given image using known encodings in the database.
    Assumes only one person per frame for simplicity.

    Returns:
        - matched name (if found)
        - 'no_persons_found' (if no face in frame)
        - 'unknown_person' (if face doesn't match any registered face)
    """

    # Step 1: Get face encodings from the image
    try:
        unknown_encodings = face_recognition.face_encodings(img)
    except Exception as e:
        print(f"Encoding error: {e}")
        return 'no_persons_found'

    if len(unknown_encodings) == 0:
        return 'no_persons_found'

    unknown_encoding = unknown_encodings[0]

    # Step 2: Search DB
    for filename in sorted(os.listdir(db_path)):
        if filename.endswith(".pickle"):
            path = os.path.join(db_path, filename)

            try:
                with open(path, 'rb') as file:
                    known_encoding = pickle.load(file)

                is_match = face_recognition.compare_faces([known_encoding], unknown_encoding)[0]

                if is_match:
                    return filename.replace('.pickle', '')

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    return 'unknown_person'
