from pyexpat import features

from fastapi import FastAPI
from numpy.ma.testutils import approx
from pydantic import BaseModel
import joblib


class ClientData(BaseModel):
    age: int
    income: float
    education: bool
    work: bool
    car: bool


app = FastAPI()
model = joblib.load("../models/model.pkl")


@app.post("/score")
def score(data: ClientData):
    features = [data.age, data.income, data.education, data.work, data.car]
    approved = not model.predict([features])[0].item()
    return {"approved": approved}
