from pydantic import BaseModel

class UserInput(BaseModel):
    weight_kg: float
    height_cm: float
    age_years: int
    gender: str
    activity_level: str

class FoodRecommendationResponse(BaseModel):
    name: str
    # distance: float
    calories: float
    proteins: float
    fat: float
    carbohydrate: float
