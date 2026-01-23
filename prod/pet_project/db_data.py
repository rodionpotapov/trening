from sqlalchemy import ForeignKey, String, Integer, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DataNeeded(Base):
    __tablename__ = "data_needed"

    client_id: Mapped[int] = mapped_column(Integer, primary_key=True)

    loan_grade: Mapped[str | None] = mapped_column(String(1), nullable=True)
    loan_int_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    loan_percent_income: Mapped[float | None] = mapped_column(Float, nullable=True)

    cb_person_default_on_file: Mapped[str | None] = mapped_column(String(1), nullable=True)
    cb_person_cred_hist_length: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ClientData(Base):
    __tablename__ = "data_from_client"

    client_id: Mapped[int] = mapped_column(ForeignKey("data_needed.client_id"), primary_key=True)

    person_age: Mapped[int] = mapped_column(Integer)
    person_income: Mapped[float] = mapped_column(Float)
    person_home_ownership: Mapped[str] = mapped_column(String(30))
    person_emp_length: Mapped[float | None] = mapped_column(Float, nullable=True)

    loan_intent: Mapped[str] = mapped_column(String(100))
    loan_amnt: Mapped[int] = mapped_column(Integer)

