from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import folium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREDICTION_FOLDER'] = 'static/predictions'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)

# Load the UNet model
unet_model = load_model("D:\\Capstone-A\\models_saved\\Unet\\Unet_50epochs.h5", compile=False)

class_colors = {
    'Water': '#E2A929',
    'Land (unpaved area)': '#8429F6',
    'Roads': '#6EC1E4',
    'Buildings': '#3C1098',
    'Vegetation': '#FEDD3A',
    'Unlabeled': '#9B9B9B'
}

def preprocess_image(image):
    image = image.convert('RGB').resize((256, 256))
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_segmentation(image_array):
    prediction = unet_model.predict(image_array)
    predicted_mask = np.argmax(prediction, axis=3)[0]
    return predicted_mask

def colorize_prediction(predicted_mask):
    height, width = predicted_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color in enumerate(class_colors.values()):
        rgb_color = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        color_mask[predicted_mask == idx] = rgb_color
    return color_mask

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/capture_map', methods=['POST'])
def capture_map():
    zoom = request.form.get('zoom', type=int)
    center_lat = request.form.get('center_lat', type=float)
    center_lng = request.form.get('center_lng', type=float)

    # Create Folium map and capture
    folium_map = folium.Map(location=[center_lat, center_lng], zoom_start=zoom, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri")
    map_path = os.path.join(app.config['UPLOAD_FOLDER'], "map.html")
    folium_map.save(map_path)

    options = Options()
    options.headless = True
    options.add_argument("--window-size=1200,1000")
    chrome_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=chrome_service, options=options)

    try:
        driver.get(f"file://{os.path.abspath(map_path)}")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        screenshot_path = os.path.join(app.config['UPLOAD_FOLDER'], "captured_map.png")
        driver.save_screenshot(screenshot_path)
    finally:
        driver.quit()

    # Process screenshot
    image = Image.open(screenshot_path)
    image_array = preprocess_image(image)
    predicted_mask = predict_segmentation(image_array)
    color_mask = colorize_prediction(predicted_mask)

    prediction_path = os.path.join(app.config['PREDICTION_FOLDER'], "map_prediction.png")
    overlay_path = os.path.join(app.config['PREDICTION_FOLDER'], "map_overlay.png")

    # Resize mask to match original image size
    resized_color_mask = Image.fromarray(color_mask).resize(image.size, Image.NEAREST)
    resized_color_mask.save(prediction_path)

    # Blend images
    overlay_image = Image.blend(image.convert('RGBA'), resized_color_mask.convert('RGBA'), alpha=0.6)
    overlay_image.save(overlay_path)

    return render_template(
        'results.html',
        source_image=url_for('static', filename='uploads/captured_map.png'),
        predicted_image=url_for('static', filename='predictions/map_prediction.png'),
        overlay_image=url_for('static', filename='predictions/map_overlay.png'),
        class_colors=class_colors
    )

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files or not request.files['file']:
        return "No file uploaded", 400

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process uploaded file
    image = Image.open(file_path)
    image_array = preprocess_image(image)
    predicted_mask = predict_segmentation(image_array)
    color_mask = colorize_prediction(predicted_mask)

    prediction_path = os.path.join(app.config['PREDICTION_FOLDER'], "uploaded_prediction.png")
    overlay_path = os.path.join(app.config['PREDICTION_FOLDER'], "uploaded_overlay.png")

    # Resize mask to match original image size
    resized_color_mask = Image.fromarray(color_mask).resize(image.size, Image.NEAREST)
    resized_color_mask.save(prediction_path)

    # Blend images
    overlay_image = Image.blend(image.convert('RGBA'), resized_color_mask.convert('RGBA'), alpha=0.6)
    overlay_image.save(overlay_path)

    return render_template(
        'results.html',
        source_image=url_for('static', filename=f'uploads/{file.filename}'),
        predicted_image=url_for('static', filename='predictions/uploaded_prediction.png'),
        overlay_image=url_for('static', filename='predictions/uploaded_overlay.png'),
        class_colors=class_colors
    )

if __name__ == '__main__':
    app.run(debug=True)
