from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_data import DataNeeded
from activate import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

rows = [
    DataNeeded(client_id=1,  loan_grade="A", loan_int_rate=0.08, loan_percent_income=0.10, cb_person_default_on_file="N", cb_person_cred_hist_length=10),
    DataNeeded(client_id=2,  loan_grade="B", loan_int_rate=0.11, loan_percent_income=0.14, cb_person_default_on_file="N", cb_person_cred_hist_length=7),
    DataNeeded(client_id=3,  loan_grade="C", loan_int_rate=0.14, loan_percent_income=0.22, cb_person_default_on_file="N", cb_person_cred_hist_length=5),
    DataNeeded(client_id=4,  loan_grade="D", loan_int_rate=0.19, loan_percent_income=0.32, cb_person_default_on_file="Y", cb_person_cred_hist_length=2),
    DataNeeded(client_id=5,  loan_grade="E", loan_int_rate=0.24, loan_percent_income=0.40, cb_person_default_on_file="Y", cb_person_cred_hist_length=1),
    DataNeeded(client_id=6,  loan_grade="B", loan_int_rate=0.10, loan_percent_income=0.12, cb_person_default_on_file="N", cb_person_cred_hist_length=9),
    DataNeeded(client_id=7,  loan_grade="C", loan_int_rate=0.16, loan_percent_income=0.26, cb_person_default_on_file="N", cb_person_cred_hist_length=3),
    DataNeeded(client_id=8,  loan_grade="D", loan_int_rate=0.21, loan_percent_income=0.35, cb_person_default_on_file="Y", cb_person_cred_hist_length=2),
    DataNeeded(client_id=9,  loan_grade="A", loan_int_rate=0.07, loan_percent_income=0.09, cb_person_default_on_file="N", cb_person_cred_hist_length=12),
    DataNeeded(client_id=10, loan_grade="E", loan_int_rate=0.26, loan_percent_income=0.45, cb_person_default_on_file="Y", cb_person_cred_hist_length=1),
]

with SessionLocal() as session:
    session.query(DataNeeded).delete()
    session.add_all(rows)
    session.commit()

print("Добавлено записей в data_needed:", len(rows))
print("client_id:", [r.client_id for r in rows])