import os
import random
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "./static/Images/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load sample dataset (limited to 500 images for demonstration)
SAMPLE_IMAGES_PATH = "./data/test_data_v2"
if not os.path.exists(SAMPLE_IMAGES_PATH):
    os.makedirs(SAMPLE_IMAGES_PATH)

all_images = [f for f in os.listdir(SAMPLE_IMAGES_PATH) if f.endswith(tuple(ALLOWED_EXTENSIONS))]
if len(all_images) > 500:
    sample_images = random.sample(all_images, 500)
else:
    sample_images = all_images


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features


image_features_dict = {}
for img_name in sample_images:
    img_path = os.path.join(SAMPLE_IMAGES_PATH, img_name)
    image_features_dict[img_name] = compute_image_features(img_path)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query_text = request.form.get("query")
        if not query_text:
            return redirect(url_for("index"))

        # Compute query features
        query_inputs = processor(text=query_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            query_features = model.get_text_features(**query_inputs)

        similarities = {}
        for img_name, img_features in image_features_dict.items():
            similarity = torch.nn.functional.cosine_similarity(query_features, img_features)
            similarities[img_name] = similarity.item()
        K = 5
        top_images = sorted(similarities, key=similarities.get, reverse=True)[:K]

        return render_template("results.html", query=query_text, images=top_images)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
