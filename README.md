# NutriLens ML-API🍱

## Overview

This project implements a Machine Learning API using FastAPI, providing two main endpoints: one for predicting food names from images and another for recommending foods based on user input.

## Endpoints

### 1. Predict Food Images

![Predict-Food-Images](https://github.com/NutriLensAI/ML-API/blob/main/images/Predict.png)

- **Endpoint**: `/predict-image`
- **Method**: POST
- **Description**: Accepts an image of food as input and returns the predicted food name.
- **Request Body**: Form-data or multipart request with the food image file.
- **Response**: JSON format containing the predicted food name.

### 2. Food Recommender System

![Recommender-System](https://github.com/NutriLensAI/ML-API/blob/main/images/Recommender.png)

- **Endpoint**: `/show-recommended-foods`
- **Method**: POST
- **Description**: Recommends 10 foods based on user input parameters: weight, height, age, gender, and activity level.
- **Request Body**: JSON format with the following fields:
  - `weight_kg`: Weight in kilograms (float)
  - `height_cm`: Height in centimeters (float)
  - `age_years`: Age in years (integer)
  - `gender`: Gender (string: 'male' or 'female')
  - `activity_level`: Activity level (string: 'sedentary', 'active', or 'very active')
- **Response**: JSON format containing a list of 10 recommended foods, each with the following details:
  - `food_name`: Name of the food (string)
  - `calories`: Calories per serving (float)
  - `proteins`: Proteins per serving (float)
  - `fat`: Fat per serving (float)
  - `carbohydrate`: Carbohydrates per serving (float)

## Usage Example

### Predict Food Images Endpoint

```bash
curl -X POST -F "file=@food_image.jpg" http://localhost:8000/predict-food-image

Response:

{
  "predicted_food": "Pizza"
}

### Recommender System Endpoint
curl -X POST -H "Content-Type: application/json" -d '{
  "weight_kg": 70.5,
  "height_cm": 175.0,
  "age_years": 30,
  "gender": "male",
  "activity_level": "active"
}' http://localhost:8000/food-recommender

Response:
{
  "recommended_foods": [
    {
      "food_name": "Salmon",
      "calories": 250,
      "proteins": 20,
      "fat": 15,
      "carbohydrate": 5
    },
    {
      "food_name": "Chicken Breast",
      "calories": 200,
      "proteins": 25,
      "fat": 8,
      "carbohydrate": 0
    },
    // More recommended foods...
  ]
}
```

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/your_username/ml-api-project.git
```


### Install Dependencies
```bash
cd ml-api-project
pip install -r requirements.txt
```

### Start the FastAPI Server
```bash
uvicorn main:app --reload
```



