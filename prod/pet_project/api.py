import json

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, Header
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from activate import DATABASE_URL
from db_data import DataNeeded, ClientData


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

with open("meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

model = CatBoostClassifier()
model.load_model("model.cbm")

threshold = float(meta["threshold"])
features = meta["features"]
cat_cols = meta["cat_cols"]

app = FastAPI()


class ClientForm(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_amnt: int


def build_df(form_dict: dict, bank_dict: dict) -> pd.DataFrame:
    row = {c: None for c in features}
    for k, v in form_dict.items():
        row[k] = v
    for k, v in bank_dict.items():
        row[k] = v
    df = pd.DataFrame([row], columns=features)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


@app.get("/session")
def session(client_id: int):
    return {"client_id": client_id}


@app.post("/predict")
def predict(form: ClientForm, x_client_id: int = Header(..., alias="X-Client-Id")):
    client_id = int(x_client_id)
    payload = form.model_dump()

    with SessionLocal() as session:
        stmt_bank = select(DataNeeded).where(DataNeeded.client_id == client_id)
        bank_obj = session.execute(stmt_bank).scalar_one_or_none()

        if bank_obj is None:
            return {
                "status": "manual_review",
                "message": "Недостаточно данных. Вы будете перенаправлены к специалисту для уточнения деталей.❤️"
            }

        bank = {
            "loan_grade": bank_obj.loan_grade,
            "loan_int_rate": bank_obj.loan_int_rate,
            "loan_percent_income": bank_obj.loan_percent_income,
            "cb_person_default_on_file": bank_obj.cb_person_default_on_file,
            "cb_person_cred_hist_length": bank_obj.cb_person_cred_hist_length,
        }

        stmt_client = select(ClientData).where(ClientData.client_id == client_id)
        client_obj = session.execute(stmt_client).scalar_one_or_none()

        if client_obj is None:
            client_obj = ClientData(client_id=client_id, **payload)
            session.add(client_obj)
        else:
            for k, v in payload.items():
                setattr(client_obj, k, v)

        df = build_df(payload, bank)

        p = float(model.predict_proba(df)[:, 1][0])
        y = int(p >= threshold)

        bank_obj.loan_status_predicted = y

        session.commit()

    return {
        "status": "ok",
        "client_id": client_id,
        "prob_default": p,
        "threshold": threshold,
        "pred_default": y
    }