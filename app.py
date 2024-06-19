

import io

from tensorflow import keras
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont

from tensorflow import keras



model = keras.models.load_model('./models/MobileNetV5_with_new_classes_v4.h5')

labels_list = ['Ayam-Geprek', 'Ayam-Goreng', 'Bakso', 'Burger', 'Ketoprak',
       'Martabak-manis (1 Potong)', 'Mie-Goreng', 'Mie-Rebus', 'Nasi-Putih', 'Nasi-Ayam-Goreng',
       'Nasi-Goreng', 'Nasi-Padang', 'Nasi-Uduk', 'Nasi-Pecel-Lele',
       'Pecel-Sayur', 'Roti-Putih', 'Sate-Ayam', 'Soto', 'Tahu-Goreng',
       'Tahu-Telur', 'Telur-Ceplok', 'Tempe-Goreng', 'Ayam-kentucky']





def preprocess_image(file_content: bytes):
    # Convert file content to BytesIO
    image_stream = io.BytesIO(file_content)

    #Load the image
    img = keras.preprocessing.image.load_img(image_stream, target_size=(224,224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Reload the original image for annotation
    image_stream.seek(0)
    original_image = Image.open(image_stream)

    return img_array, original_image

def predict_image(img_array: np.ndarray):
    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = labels_list[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    return predicted_class, confidence

def draw_bounding_box(image: Image, predicted_class: str, confidence:float):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Load a larger font
    font_size = 30  # Increase this value to make the text larger
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Ensure the font file is available on your system
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if TTF font is not available

    #define bounding box coordinates (sementara pake entire image)
    box_coords = [(10, 10), (image.size[0] - 10, image.size[1] - 10)]

    # Draw Bounding Box
    draw.rectangle(box_coords, outline="red", width=5)

    # Draw text
    text = f"{predicted_class}: {confidence:.2f}%"
    text_location = (10, 10)
    text_bbox = draw.textbbox(text_location, text, font=font)
    padding = 5

    # Increase the bounding box size by adding padding
    padded_bbox = [
        (text_bbox[0] - padding, text_bbox[1] - padding),
        (text_bbox[2] + padding, text_bbox[3] + padding)
    ]

    draw.rectangle(padded_bbox, fill="red")
    draw.text(text_location, text, fill="white", font=font)

    # Save annotated image to a BytesIO object
    annotated_image_stream = io.BytesIO()
    image.save(annotated_image_stream, format="JPEG")
    annotated_image_stream.seek(0)

    return annotated_image_stream