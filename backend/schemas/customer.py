from pydantic import BaseModel, Field
from typing import Optional

class CustomerInput(BaseModel):
    Year_Birth: int = Field(default=1985, description="Birth year of customer, e.g., 1985", ge=1920, le=2015)
    Education: str = Field(default="Graduation", description="Education level: Graduation, Master, PhD, Basic, 2n Cycle")
    Marital_Status: str = Field(default="Single", description="Marital status: Single, Married, Together, Divorced, Widow")
    Income: Optional[float] = Field(default=50000.0, description="Annual income of customer")
    Kidhome: int = Field(default=0, description="Number of young children in household", ge=0, le=5)
    Teenhome: int = Field(default=0, description="Number of teenagers in household", ge=0, le=5)
    Recency: float = Field(default=45.0, description="Days since last purchase", ge=0, le=365)
    Customer_Tenure_Days: int = Field(default=1000, description="Customer relationship tenure in days", ge=0, le=5000)
    
    # Spending components
    MntWines: float = Field(default=200.0, description="Amount spent on Wine", ge=0)
    MntFruits: float = Field(default=30.0, description="Amount spent on Fruits", ge=0)
    MntMeatProducts: float = Field(default=150.0, description="Amount spent on Meat Products", ge=0)
    MntFishProducts: float = Field(default=40.0, description="Amount spent on Fish Products", ge=0)
    MntSweetProducts: float = Field(default=20.0, description="Amount spent on Sweet Products", ge=0)
    MntGoldProds: float = Field(default=30.0, description="Amount spent on Gold Products", ge=0)
    
    # Purchase Channels & Activity
    NumDealsPurchases: int = Field(default=2, description="Number of purchases made with a discount", ge=0)
    NumWebPurchases: int = Field(default=4, description="Number of purchases made through website", ge=0)
    NumCatalogPurchases: int = Field(default=2, description="Number of purchases made using catalog", ge=0)
    NumStorePurchases: int = Field(default=6, description="Number of purchases made directly in stores", ge=0)
    NumWebVisitsMonth: int = Field(default=5, description="Number of visits to company website in last month", ge=0)
    
    # Campaigns & Complaints
    Complain: int = Field(default=0, description="1 if customer complained in last 2 years, 0 otherwise", ge=0, le=1)
    Response: int = Field(default=0, description="1 if customer accepted offer in last campaign, 0 otherwise", ge=0, le=1)

class CustomerPredictionResponse(BaseModel):
    cluster: int
    cluster_name: str
    message: str
    summary: dict
    pca_coordinates: list[float]
