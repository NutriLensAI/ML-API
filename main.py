
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import numpy as np
import io
import os

import keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img

from app import predict_image
from app import preprocess_image
from app import draw_bounding_box

app_desc = """<h2>Work In Progress</h2>
<h2>sementara baru endpoint predict aja</h2>

List Thing to create:
1. Endpoint Uploaded Image with bbox (Bounding Box) contains info of predicted images (DONE)
2. Endpoint showing nutrition
3. Endpoint buat Recommender System (blm terlalu paham perlu re discuss lg sm anak2)
<br>by NutriLens"""

app = FastAPI(title='NutriLens ML API Endpoint', description=app_desc)



# model = keras.models.load_model('./models/model_MobileNetV5.h5')

# labels_list = ["Ayam-Goreng", "Ayam-goreng-dada", "Ayam-goreng-paha", "Bakso", "Bit", "Burger",
#                "Coto-mangkasara-kuda-masakan", "Ketoprak", "Mie-Goreng", "Mie-basah", "Nasi",
#                "Nasi-Goreng", "Nasi-Padang", "Nasi-Uduk", "Roti-Bakar", "Roti-Putih", "Sate-Ayam",
#                "Sate-Kambing", "Soto-betawi-masakan", "Tahu", "Tahu Telur", "Telur-Ceplok",
#                "Tempe-Goreng", "soto-banjar-masakan"]


# async def predict_api(file: UploadFile = File(...)):
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#         return "Image must be jpg or png format!"
#     image = read_imagefile(await file.read())
#     prediction = predict


# @app.post('/predict-image')
# async def predict_image(file: UploadFile = File(...)):
#     # Load the uploaded image file
#     file_content = await file.read()
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#      return "Image must be jpg or png format!"
#
#     image_stream = io.BytesIO(file_content)
#     img = keras.preprocessing.image.load_img(image_stream, target_size=(224, 224))  # Adjust target_size according to your model input
#     img = keras.preprocessing.image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img / 255.0
#     # Make prediction
#     predictions = model.predict(img)
#     predicted_class = labels_list[np.argmax(predictions[0])]
#
#     return JSONResponse(content={"prediction": predicted_class})
#

@app.post('/predict-image')
async def predict_image_to_json(file: UploadFile = File(...)):
    #Load the uploaded image file
    file_content = await file.read()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return JSONResponse(content={"error": "Image must be jpg or png format!"}, status_code=400)

    #preprocess the image
    img_array, original_image = preprocess_image(file_content)

    #Predict the image
    predicted_class, confidence = predict_image(img_array)

    #Draw bounding box on the original image
    annotated_image_stream = draw_bounding_box(original_image, predicted_class, confidence)

    # Ensure the temp directory exists
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    #Save the uploaded image temporarily
    file_path = f"./temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(annotated_image_stream.getvalue())

    # Create HTML response
    html_content = f"""
        <html>
            <body>
                <h1>Prediction: {predicted_class} ({confidence:.2f}%</h1>
                <img src="/temp/{file.filename}" alt="Uploaded Image" />
            </body>
        </html>
        """

    return HTMLResponse(content=html_content)

    #{"prediction: ": predicted_class}

@app.get("/temp/{filename}")
async def get_image(filename: str):
    file_path = f"./temp/{filename}"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return HTMLResponse(content=f.read(), media_type="image/jpeg")
    return JSONResponse(content={"error": "File not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
