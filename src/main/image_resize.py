from PIL import Image
import os

folder = "./dataset"
output = "./resized"

os.makedirs(output, exist_ok=True)

for img_file in os.listdir(folder):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        img = Image.open(os.path.join(folder, img_file))
        img = img.resize((512, 512), Image.LANCZOS)
        img.save(os.path.join(output, img_file))