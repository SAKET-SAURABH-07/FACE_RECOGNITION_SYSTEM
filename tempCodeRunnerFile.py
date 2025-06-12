import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import util
from face_test import test  # ✅ renamed to avoid conflict with Python's built-in test package


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        self.main_window.title("Face Attendance System")

        # Buttons
        self.login_button_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'Logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'Register New User', 'gray',
            self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        # Webcam feed
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)
        self.add_webcam(self.webcam_label)

        # Paths
        self.db_dir = './db'
        os.makedirs(self.db_dir, exist_ok=True)

        self.log_path = './log.txt'

        # Anti-spoof model path (✅ make this customizable)
        self.spoof_model_path = './resources/anti_spoof_models'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.most_recent_capture_arr = frame
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def handle_auth(self, is_login=True):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir=self.spoof_model_path,
            device_id=0
        )

        if label != 1:
            util.msg_box('Spoof Detected!', 'You are not a real person!')
            return

        name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Unknown User', 'Please register first or try again.')
        else:
            action = 'in' if is_login else 'out'
            message = f"Welcome, {name}!" if is_login else f"Goodbye, {name}!"
            util.msg_box('Authentication Successful', message)
            with open(self.log_path, 'a') as f:
                f.write(f"{name},{datetime.datetime.now()},{action}\n")

    def login(self):
        self.handle_auth(is_login=True)

    def logout(self):
        self.handle_auth(is_login=False)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        self.register_new_user_window.title("Register New User")

        # UI
        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try Again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Please,\ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()

        if not name:
            util.msg_box("Error", "Name cannot be empty!")
            return

        try:
            embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
        except IndexError:
            util.msg_box("Error", "No face detected! Try again.")
            return

        with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as file:
            pickle.dump(embeddings, file)

        util.msg_box('Success!', f'User "{name}" registered successfully!')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
