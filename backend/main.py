from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import prediction, analytics
from backend.services.ml_service import ml_service

app = FastAPI(
    title="SmartCart Clustering System API",
    description="Backend service for Customer Segmentation using Unsupervised Machine Learning",
    version="1.0.0"
)

# Enable CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root API Status Endpoint
@app.get("/")
def api_root():
    return {
        "message": "SmartCart Clustering API",
        "status": "running"
    }

# Health Check Endpoint
@app.get("/health")
def health_check():
    is_ml_ready = ml_service.is_healthy()
    return {
        "status": "healthy" if is_ml_ready else "unhealthy",
        "backend": "online",
        "ml_service": "loaded" if is_ml_ready else "not_loaded",
        "dataset_records": len(ml_service.df_clustered) if is_ml_ready else 0
    }

# Include API routes
app.include_router(prediction.router)
app.include_router(analytics.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
