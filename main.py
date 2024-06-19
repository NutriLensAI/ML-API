
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import pandas as pd
from typing import List
from tensorflow import keras
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array, load_img

from utils.predict import predict_image
from utils.predict import preprocess_image

from utils.Recommender import NutritionCalculator
from models import UserInput, FoodRecommendationResponse


#
# with open('recommender_model.pkl', 'rb') as file:
#     recommenderModel = pickle.load(file)


app_desc = """
<h2>ML API Endpoint</h2>

1. Predict Image
2. Food Recommendation System

<br>by NutriLens"""

app = FastAPI(title='NutriLens ML API Endpoint', description=app_desc)



model = keras.models.load_model('models/MobileNetV5_with_new_classes_v4.h5')

labels_list = ['Ayam-Geprek', 'Ayam-Goreng', 'Bakso', 'Burger', 'Ketoprak',
       'Martabak-manis (1 Potong)', 'Mie-Goreng', 'Mie-Rebus', 'Nasi-Putih', 'Nasi-Ayam-Goreng',
       'Nasi-Goreng', 'Nasi-Padang', 'Nasi-Uduk', 'Nasi-Pecel-Lele',
       'Pecel-Sayur', 'Roti-Putih', 'Sate-Ayam', 'Soto', 'Tahu-Goreng',
       'Tahu-Telur', 'Telur-Ceplok', 'Tempe-Goreng', 'Ayam-kentucky']


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

    # #Draw bounding box on the original image
    # annotated_image_stream = draw_bounding_box(original_image, predicted_class, confidence)

    # # Ensure the temp directory exists
    # temp_dir = "./temp"
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    #
    # #Save the uploaded image temporarily
    # file_path = f"./temp/{file.filename}"
    # with open(file_path, "wb") as f:
    #     f.write(annotated_image_stream.getvalue())

    return {"prediction: ": predicted_class,
            "confidence: " : confidence
            }

# @app.get("/temp/{filename}")
# async def get_image(filename: str):
#     file_path = f"./temp/{filename}"
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             return HTMLResponse(content=f.read(), media_type="image/jpeg")
#     return JSONResponse(content={"error": "File not found"}, status_code=404)



#THE RECOMMENDER SYSTEM
# Load the dataset
df = pd.read_csv("./data/nutrition.csv")  # Update this with the actual path to your CSV file

@app.post("/show-recommended-foods", response_model=List[FoodRecommendationResponse])
def show_recommended_foods(input_data: UserInput):
    try:
        # Extract input data
        weight_kg = input_data.weight_kg
        height_cm = input_data.height_cm
        age_years = input_data.age_years
        gender = input_data.gender
        activity_level = input_data.activity_level

        # Kalkulasi BMR
        bmr = NutritionCalculator.calculate_bmr(weight_kg, height_cm, age_years, gender)

        # Level Aktivitas
        activity_factor = NutritionCalculator.get_activity_factor(activity_level)

        # Total Kalori harian : 3 (asumsi makan 1 hari 3 kali)
        total_calories = (bmr * activity_factor) / 10

        # Kalkulasi mikro nutrisi dalam satu kali makan
        protein_grams = np.round((total_calories * 0.50) / 4,1)  # 50% protein
        carbohydrate_grams = np.round((total_calories * 0.20) / 9,1)  # 20% carbohydrates
        fat_grams = np.round((total_calories * 0.30) / 4,1)  # 30% fat

        # Buat kebutuhan nutrisi
        user_preferences = np.array([total_calories, protein_grams, fat_grams, carbohydrate_grams])

        recommended_foods = NutritionCalculator.recommended_food(user_preferences, df)

        response = [
            FoodRecommendationResponse(
                name=food[0],
                # distance=food[1],
                calories=food[2],
                proteins=food[3],
                fat=food[4],
                carbohydrate=food[5]
            ) for food in recommended_foods[:10]
        ]

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
