# Bringing in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()


class ScoringItem(BaseModel):
    YearsAtCompany: float  # 1, // Float Value
    EmployeeSatisfaction: float  # 0.01, // Float Value
    Position: str  # 'Non-Manager', // Manager or Non-Manager
    Salary: int  # 4.0 // Ordinal 1,2,3,4,5


with open('rfmodel.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post("/")
async def scoring_endpoint(item: ScoringItem):
    df = pd.DataFrame([dict(item).values()], columns=dict(item).keys())
    yhat = model.predict(df)
    return {"prediction": int(yhat)}
