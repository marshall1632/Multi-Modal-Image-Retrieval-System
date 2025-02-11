# Multi-Modal Image Retrieval System

This project implements a Multi-Modal Image Retrieval System using Flask, CLIP (Contrastive Language–Image Pretraining) model from OpenAI, and a dataset of images. The system allows users to input a text query and retrieve the most relevant images based on their similarity to the query.

## Dataset

The dataset used for this project is from Kaggle: [AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data?select=test_data_v2).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/multi-modal-image-retrieval.git
    cd multi-modal-image-retrieval
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset from Kaggle and place the `test_data_v2` folder inside a `data` directory in the project root.

## Configuration

Set the upload folder and allowed extensions for image uploads in the Flask app configuration:
```python
UPLOAD_FOLDER = "./static/Images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

## Running the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## Usage

1. On the home page, enter a text query in the input field and submit the form.
2. The application will process the query, compute the similarities between the query and the images in the dataset, and display the top 5 most similar images.

---

## Code Overview

### `app.py`
- **Main Application File:** Sets up the Flask app and defines the routes and functionality.
- **Imports:** Necessary libraries and modules are imported, including Flask, PIL, transformers, and torch.
- **App Configuration:** Configures the Flask app with an upload folder and allowed file extensions.
- **Model and Processor Initialization:** Initializes the CLIP model and processor from OpenAI.
- **Dataset Loading:** Loads the dataset and uses a sample of 500 images for demonstration.
- **Helper Functions:** Defines functions for checking allowed file types and computing image features.
- **Image Features Computation:** Computes features for the sample images and stores them in a dictionary.
- **Routes:**
  - `/`: Handles both GET and POST requests. On POST, it processes the query, computes similarities, and returns the top 5 images.

### `templates/index.html`
- The home page template where users can input their text queries.

### `templates/results.html`
- The results page template that displays the query and the top 5 most similar images.

---

## Dependencies

- Flask
- Pillow
- transformers
- torch

---

## Notes

- Ensure you have a stable internet connection as the CLIP model and processor are loaded from the Hugging Face model hub.
- The application does not include CSS for styling, focusing purely on functionality.

---

## Contributing

Feel free to fork this repository and contribute improvements or new features. Pull requests are welcome!


## Acknowledgments

- [OpenAI](https://openai.com/) for the CLIP model.
- [Hugging Face](https://huggingface.co/) for hosting the CLIP model and processor.
- [Kaggle](https://www.kaggle.com/) for providing the dataset.
```

You can copy and paste this directly into your `README.md` file. Let me know if you need any further adjustments!
