import requests
import streamlit as st
import random 
API_URL = "http://127.0.0.1:8000/predict"

st.title("Форма для подбора кредита")

if "client_id" not in st.session_state:
    st.session_state.client_id = random.randint(1, 10)

client_id = st.session_state.client_id

home_map = {
    "Арендую": "RENT",
    "Ипотека": "MORTGAGE",
    "Собственность": "OWN",
    "Живу у родственников": "OTHER",
}

intent_map = {
    "Образование": "EDUCATION",
    "Медицина": "MEDICAL",
    "Покупка": "PERSONAL",
    "Ремонт": "HOMEIMPROVEMENT",
    "Погашение долгов": "DEBTCONSOLIDATION",
    "Бизнес": "VENTURE",
}

with st.form("Заполните нужные данные❤️"):
    # client_id = st.number_input("Номер клиента", min_value=1, step=1)

    person_age = st.number_input("Возраст", min_value=18, step=1)
    person_income = st.number_input("Доход в тыс-руб/мес.", min_value=0.0)

    person_home_ownership_ru = st.selectbox("Жильё", list(home_map.keys()))
    person_emp_length = st.number_input("Стаж работы (лет)", min_value=0)

    loan_intent_ru = st.selectbox("Цель кредита", list(intent_map.keys()))
    loan_amnt = st.number_input("Сумма кредита", min_value=0)

    submitted = st.form_submit_button("Отправить")

if submitted:
    payload = {
        "person_age": int(person_age),
        "person_income": float(person_income * 12 / 80),
        "person_home_ownership": home_map[person_home_ownership_ru],
        "person_emp_length": float(person_emp_length),
        "loan_intent": intent_map[loan_intent_ru],
        "loan_amnt": int(loan_amnt / 80),
    }

    headers = {"X-Client-Id": str(int(client_id))}

    r = requests.post(API_URL, json=payload, headers=headers, timeout=10)

    if r.status_code != 200:
        st.error(f"Ошибка API: {r.status_code} {r.text}")
    else:
        resp = r.json()
        if resp.get("status") == "manual_review":
            st.warning("Недостаточно данных. Заявка отправлена на дополнительную проверку специалистом.")
        else:
            if resp["pred_default"] == 1:
                st.error("отказано")
                st.write(f"Вероятность дефолта: {resp['prob_default']:.4f}")
            else:
                st.success("Поздравляю, заявка предварительно одобрена❤️")