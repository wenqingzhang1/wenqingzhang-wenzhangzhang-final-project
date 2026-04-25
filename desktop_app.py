import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog

# ==========================================
# 1. Initialization settings and database feature preloading
# ==========================================
# Set UI theme and color style
ctk.set_appearance_mode("Dark")  # Dark mode
ctk.set_default_color_theme("blue")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(pth_path):
    """Load your trained model (same logic as before)"""
    print("[*] Loading model weights...")
    # TODO: Replace with your actual model construction and loading code
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

# model = load_model("./Step10_exp_sop_resnet50_v2/model_epoch1.pth.tar")
model = load_model("./Step10_exp_sop_resnet50_v2/model_best.pth.tar")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DATABASE_DIR = "./Matching base"  # Test image database folder
db_image_paths = []
db_features = []

print("[*] Building retrieval database feature set...")
if os.path.exists(DATABASE_DIR):
    for img_path in glob.glob(os.path.join(DATABASE_DIR, "*.*")):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = F.normalize(model(img_tensor), p=2, dim=1)
            db_features.append(feat.squeeze())
            db_image_paths.append(img_path)
        except Exception:
            continue

if db_features:
    db_features_tensor = torch.stack(db_features).to(device)
else:
    print("[!] Warning: test image database is empty!")

# ==========================================
# 2. Build desktop GUI application
# ==========================================
class ImageRetrievalApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window basic settings
        self.title("SOP Image Retrieval System - Graduation Project Demo")
        self.geometry("1000x700")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- Left: Control Panel ----------------
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Visual Retrieval System",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Upload button
        self.upload_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="📂 Select Query Image",
            command=self.load_query_image
        )
        self.upload_btn.grid(row=1, column=0, padx=20, pady=10)

        # Top-K slider
        self.slider_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Number of Results (Top-K): 5"
        )
        self.slider_label.grid(row=2, column=0, padx=20, pady=(10, 0))

        self.top_k_slider = ctk.CTkSlider(
            self.sidebar_frame,
            from_=1,
            to=20,
            number_of_steps=19,
            command=self.update_slider_label
        )
        self.top_k_slider.set(5)
        self.top_k_slider.grid(row=3, column=0, padx=20, pady=10)

        # Display selected query image
        self.query_image_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="No image selected"
        )
        self.query_image_label.grid(row=4, column=0, padx=20, pady=20, sticky="n")

        # ---------------- Right: Retrieval Results Gallery ----------------
        self.results_frame = ctk.CTkScrollableFrame(
            self,
            label_text="Top Matches Retrieval Results"
        )
        self.results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure grid layout (3 columns)
        self.results_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Store displayed images to prevent garbage collection
        self.result_images = []

    def update_slider_label(self, value):
        self.slider_label.configure(
            text=f"Number of Results (Top-K): {int(value)}"
        )

    def load_query_image(self):
        """Open file dialog, load image and trigger retrieval"""
        file_path = filedialog.askopenfilename(
            title="Select a product image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG")]
        )
        if not file_path:
            return

        # 1. Display selected image on left panel
        img_pil = Image.open(file_path).convert("RGB")
        ui_img = ctk.CTkImage(
            light_image=img_pil,
            dark_image=img_pil,
            size=(200, 200)
        )
        self.query_image_label.configure(image=ui_img, text="")
        self.query_image_label.image = ui_img  # keep reference

        # 2. Run retrieval
        self.run_retrieval(img_pil)

    def run_retrieval(self, img_pil):
        """Compute features and update right panel"""
        if len(db_features) == 0:
            return

        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.result_images.clear()

        # Extract features and compute similarity
        top_k = int(self.top_k_slider.get())
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            query_feat = F.normalize(model(img_tensor), p=2, dim=1).squeeze()

        similarities = torch.matmul(db_features_tensor, query_feat)
        top_k = min(top_k, len(db_image_paths))
        scores, indices = torch.topk(similarities, k=top_k)

        # Display results
        for i, (score, idx) in enumerate(zip(scores, indices)):
            img_path = db_image_paths[idx.item()]

            # Resize result image for UI
            res_img_pil = Image.open(img_path).convert("RGB")
            res_ui_img = ctk.CTkImage(
                light_image=res_img_pil,
                dark_image=res_img_pil,
                size=(180, 180)
            )
            self.result_images.append(res_ui_img)

            # Create card
            card = ctk.CTkFrame(self.results_frame)

            # Row/column (3 per row)
            row = i // 3
            col = i % 3
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            # Image
            img_label = ctk.CTkLabel(card, image=res_ui_img, text="")
            img_label.pack(padx=10, pady=(10, 5))

            # Score text
            score_text = f"Top {i+1} | Similarity: {score.item():.4f}"
            text_label = ctk.CTkLabel(
                card,
                text=score_text,
                font=ctk.CTkFont(size=12)
            )
            text_label.pack(padx=10, pady=(0, 10))

if __name__ == "__main__":
    app = ImageRetrievalApp()
    app.mainloop()