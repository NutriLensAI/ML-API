import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

class NutritionCalculator:
    @staticmethod
    def calculate_bmr(weight_kg, height_cm, age_years, gender):
        if gender.lower() == "male":
            bmr = 66 + (13.7 * weight_kg) + (5 * height_cm) - (6.8 * age_years)
        elif gender.lower() == "female":
            bmr = 655 + (9.6 * weight_kg) + (1.8 * height_cm) - (4.7 * age_years)
        else:
            raise ValueError("Invalid gender. Please enter 'male' or 'female'.")
        return bmr

    @staticmethod
    def get_activity_factor(activity_level):
        activity_factor_map = {
            "sedentary": 1.0,
            "moderately active": 1.375,
            "active": 1.55,
            "very active": 1.725
        }
        activity_level = activity_level.lower()
        if activity_level in activity_factor_map:
            return activity_factor_map[activity_level]
        else:
            raise ValueError("Invalid activity level. Please enter one of the following: sedentary, moderately active, active, very active.")

    @staticmethod
    def recommended_food(user_preferences, df):
        distances = []
        for index, row in df.iterrows():
            food_nutrition = np.array([row['calories'], row['proteins'], row['fat'], row['carbohydrate']])
            distance = euclidean(user_preferences, food_nutrition)
            distances.append((row['name'], distance, row['calories'], row['proteins'], row['fat'], row['carbohydrate']))
        distances.sort(key=lambda x: x[1])
        return distances

