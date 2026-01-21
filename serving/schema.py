from pydantic import BaseModel


class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0
    PaymentMethod_Credit_card_automatic: int = 0


class PredictionResponse(BaseModel):
    churn_probability: float
