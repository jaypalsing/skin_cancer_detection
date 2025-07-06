import os
import sys
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import albumentations as A
from albumentations.pytorch import ToTensorV2

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# Paths and setup
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

from src.model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Transforms
# ==========================================
def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

transforms = get_transforms()

# ==========================================
# Load Model
# ==========================================
def load_model(model_name):
    checkpoint_path = os.path.join(BASE_DIR, "outputs", "checkpoints", f"{model_name}_final.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = get_model(model_name=model_name).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model

# ==========================================
# Prediction
# ==========================================
def predict(model, image_pil):
    img_array = np.array(image_pil)
    transformed = transforms(image=img_array)['image']
    transformed = transformed.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(transformed)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        prob = probs[0, pred_idx].item()

    return pred_idx, prob, probs[0].cpu().numpy()

# ==========================================
# GUI Application
# ==========================================
class SkinCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Skin Cancer Detection")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        self.root.configure(bg="#f0f0f0")

        # Main frame: 2 columns
        self.main_frame = tk.Frame(root, bg="#f0f0f0")
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=0)
        self.main_frame.rowconfigure(1, weight=1)

        # Left controls
        self.left_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.left_frame.grid(row=0, column=0, sticky="nwe", padx=10, pady=10)

        title_label = tk.Label(
            self.left_frame, text="Skin Cancer Detection",
            font=("Helvetica", 18, "bold"),
            bg="#f0f0f0", fg="#333"
        )
        title_label.pack(pady=10)

        # Model selector
        model_frame = tk.Frame(self.left_frame, bg="#f0f0f0")
        model_frame.pack(pady=5)

        tk.Label(
            model_frame, text="Select Model:",
            font=("Arial", 12), bg="#f0f0f0"
        ).pack(side=tk.LEFT, padx=5)

        self.model_var = tk.StringVar(value="resnet50")
        self.model_selector = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=["resnet50", "densenet121"],
            state="readonly",
            width=15
        )
        self.model_selector.pack(side=tk.LEFT, padx=5)

        # Upload button
        self.upload_button = tk.Button(
            self.left_frame, text="Browse Image",
            font=("Arial", 12),
            bg="#4caf50", fg="white",
            command=self.load_image
        )
        self.upload_button.pack(pady=5)

        self.filename_label = tk.Label(
            self.left_frame, text="No file selected.",
            font=("Arial", 10),
            bg="#f0f0f0", fg="#666"
        )
        self.filename_label.pack(pady=2)

        # Predict button
        self.predict_button = tk.Button(
            self.left_frame, text="Predict",
            font=("Arial", 12),
            bg="#2196f3", fg="white",
            state=tk.DISABLED,
            command=self.run_prediction
        )
        self.predict_button.pack(pady=10)

        # Image display
        self.image_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.image_label = tk.Label(self.image_frame, bg="#ddd")
        self.image_label.pack(pady=10)

        # Results & graph
        self.right_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.result_label = tk.Label(
            self.right_frame, text="",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        self.result_label.pack(pady=10)

        self.progress = ttk.Progressbar(
            self.right_frame, orient="horizontal", length=300, mode="determinate"
        )
        self.progress.pack(pady=10)

        self.graph_frame = tk.Frame(self.right_frame, bg="#f0f0f0")
        self.graph_frame.pack(pady=10, fill="both", expand=True)

        self.image_pil = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return
        self.image_pil = Image.open(file_path).convert("RGB")
        img_resized = self.image_pil.resize((400, 400))
        tk_image = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

        filename = os.path.basename(file_path)
        self.filename_label.config(text=f"Selected: {filename}")
        self.result_label.config(text="")
        self.predict_button.config(state=tk.NORMAL)
        self.progress["value"] = 0

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

    def run_prediction(self):
        if self.image_pil is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        model_name = self.model_var.get()
        try:
            model = load_model(model_name)
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            return

        pred_idx, prob, probs = predict(model, self.image_pil)

        # Determine the max probability and the class index
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]
        label = "Melanoma" if max_idx == 1 else "Benign"

        # If uncertain prediction
        if max_prob < 0.8:
            self.result_label.config(
                text="⚠️ Uncertain – Image may not be a valid skin lesion.",
                fg="#ff8800"
            )
            self.progress["value"] = 0
        else:
            self.result_label.config(
                text=f"Prediction: {label}\nProbability: {max_prob:.2%}",
                fg="#d32f2f" if label == "Melanoma" else "#388e3c"
            )
            self.progress["value"] = int(max_prob * 100)

        self.root.update_idletasks()

        # Create the graph
        fig = Figure(figsize=(4, 2.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(["Benign", "Melanoma"], probs, color=["#4caf50", "#d32f2f"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        for i, v in enumerate(probs):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=10)

        # Clear and display graph
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ==========================================
# Run the app
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SkinCancerApp(root)
    root.mainloop()
