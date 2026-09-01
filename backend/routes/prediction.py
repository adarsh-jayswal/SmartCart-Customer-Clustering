from fastapi import APIRouter, HTTPException, status
from backend.schemas.customer import CustomerInput, CustomerPredictionResponse
from backend.services.ml_service import ml_service

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/predict", response_model=CustomerPredictionResponse)
def predict_customer(input_data: CustomerInput):
    try:
        raw_dict = input_data.model_dump()
        result = ml_service.predict(raw_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing cluster prediction: {str(e)}"
        )
