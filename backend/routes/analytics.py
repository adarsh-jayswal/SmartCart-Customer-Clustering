from fastapi import APIRouter, HTTPException, status, UploadFile, File, Response
from fastapi.responses import StreamingResponse
import io
from backend.services.ml_service import ml_service

router = APIRouter(prefix="/api", tags=["Analytics & Dataset Clustering"])

@router.post("/cluster")
async def cluster_dataset(file: UploadFile = File(None)):
    """
    Executes customer clustering on uploaded CSV dataset (or default dataset if no file uploaded).
    """
    try:
        if file is not None and file.filename != "":
            contents = await file.read()
            result = ml_service.process_csv_file(contents)
        else:
            result = ml_service.get_last_result_response()
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing customer dataset clustering: {str(e)}"
        )

@router.get("/download-clustered-csv")
def download_clustered_csv():
    """
    Downloads the clustered dataset as CSV containing original records + Cluster column.
    """
    try:
        csv_bytes = ml_service.generate_clustered_csv()
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=smartcart_clustered_customers.csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating clustered CSV file: {str(e)}"
        )

@router.get("/analytics")
def get_analytics():
    try:
        return ml_service.get_analytics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dataset analytics: {str(e)}"
        )

@router.get("/clusters")
def get_clusters():
    try:
        clusters = ml_service.get_clusters_info()
        return {"clusters": clusters, "total_clusters": len(clusters)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching cluster details: {str(e)}"
        )
