from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_data import *

DATABASE_URL = 'postgresql+psycopg2://postgres:1703@localhost:5432/clients'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(engine)


def create_db_tables():
    Base.metadata.create_all(engine)


create_db_tables()
