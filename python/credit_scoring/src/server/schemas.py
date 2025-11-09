"""
python/credit_scoring/src/server/schemas.py: this file ensures that data entering and leaving the API has the correct structure and types.
"""

from enum import Enum
from pydantic import BaseModel, Field


class SexEnum(str, Enum):
    male = 'male'
    female = 'female'


class HousingEnum(str, Enum):
    own = 'own'
    rent = 'rent'
    free = 'free'


class SavingAccountsEnum(str, Enum):
    na = 'NA'
    little = 'little'
    moderate = 'moderate'
    quite_rich = 'quite rich'
    rich = 'rich'


class CheckingAccountEnum(str, Enum):
    na = 'NA'
    little = 'little'
    moderate = 'moderate'
    rich = 'rich'


class PurposeEnum(str, Enum):
    car = 'car'
    furniture_equipment = 'furniture/equipment'
    radio_tv = 'radio/TV'
    domestic_appliances = 'domestic appliances'
    repairs = 'repairs'
    education = 'education'
    business = 'business'
    vacation_others = 'vacation/others'
    

# input schemas
class CreditRiskInput(BaseModel):
    """
    Define the structure of the input data for prediction.
    Field names must match the columns in the original dataset.
    """
    Age: int = Field(..., gt=0, description="Edad del solicitante en años.")
    Sex: SexEnum = Field(..., description="Sexo del solicitante.")
    Job: int = Field(..., ge=0, le=3, description="Nivel de habilidad laboral (0-3).")
    Housing: HousingEnum = Field(..., description="Tipo de vivienda.")
    Saving_accounts: SavingAccountsEnum = Field(..., alias="Saving accounts", description="Estado de la cuenta de ahorros.")
    Checking_account: CheckingAccountEnum = Field(..., alias="Checking account", description="Estado de la cuenta corriente.")
    Credit_amount: float = Field(..., gt=0, alias="Credit amount", description="Monto del crédito solicitado.")
    Duration: int = Field(..., gt=0, description="Duración del crédito en meses.")
    Purpose: PurposeEnum = Field(..., description="Propósito del crédito.")
    
    
    class Config:
        allow_population_by_field_name = True  # Permite usar 'Saving accounts' en lugar de 'Saving_accounts'
        schema_extra = {
            "example": {
                "Age": 35,
                "Sex": "male",
                "Job": 1,
                "Housing": "free",
                "Saving accounts": "NA",
                "Checking account": "NA",
                "Credit amount": 9055,
                "Duration": 36,
                "Purpose": "education"
            }
        }
        

# output schemas
class CreditRiskOutput(BaseModel):
    """
    Define API response structure.
    """
    prediction: str = Field(..., description="Predicción del riesgo ('good' o 'bad').")
    probability: float = Field(..., ge=0, le=1, description="Probabilidad de que el riesgo sea 'good'.")
    
        