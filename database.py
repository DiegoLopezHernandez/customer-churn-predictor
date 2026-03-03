"""
database.py - Gestión de la base de datos SQLite con SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

# ── Tabla: Clientes ──────────────────────────────────────────────────────────
class Cliente(Base):
    __tablename__ = "clientes"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    customer_id      = Column(String(20), unique=True, nullable=False)
    gender           = Column(String(10))
    senior_citizen   = Column(Integer)
    partner          = Column(String(5))
    dependents       = Column(String(5))
    tenure           = Column(Integer)
    phone_service    = Column(String(5))
    multiple_lines   = Column(String(25))
    internet_service = Column(String(20))
    contract         = Column(String(25))
    monthly_charges  = Column(Float)
    total_charges    = Column(Float)
    churn_real       = Column(String(5))
    creado_en        = Column(DateTime, default=datetime.utcnow)

# ── Tabla: Predicciones ──────────────────────────────────────────────────────
class Prediccion(Base):
    __tablename__ = "predicciones"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    customer_id         = Column(String(20))
    probabilidad_churn  = Column(Float)
    prediccion          = Column(String(5))   # "Yes" / "No"
    modelo_usado        = Column(String(50))
    fecha_prediccion    = Column(DateTime, default=datetime.utcnow)
    notas               = Column(Text)

# ── Tabla: Métricas del modelo ───────────────────────────────────────────────
class MetricaModelo(Base):
    __tablename__ = "metricas_modelo"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    modelo     = Column(String(50))
    accuracy   = Column(Float)
    precision  = Column(Float)
    recall     = Column(Float)
    f1_score   = Column(Float)
    fecha      = Column(DateTime, default=datetime.utcnow)


# ── Setup ────────────────────────────────────────────────────────────────────
def get_engine(db_path: str = "churn.db"):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
