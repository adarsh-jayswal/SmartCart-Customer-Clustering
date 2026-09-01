import os
import io
import pickle
import pandas as pd
import numpy as np

from ml.pipeline import run_ml_pipeline, process_dataset, preprocess_single_input

ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml", "artifacts"))
ARTIFACTS_PATH = os.path.join(ARTIFACTS_DIR, "model_artifacts.pkl")
DATASET_PATH = os.path.join(ARTIFACTS_DIR, "clustered_dataset.csv")
RAW_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "smartcart_customers.csv"))

class MLService:
    def __init__(self):
        self.artifacts = None
        self.df_clustered = None
        self.last_result = None
        self.load_or_train()

    def load_or_train(self):
        if os.path.exists(RAW_DATASET_PATH):
            try:
                df_raw = pd.read_csv(RAW_DATASET_PATH)
                self.last_result = process_dataset(df_raw)
                self.artifacts = self.last_result["artifacts"]
                self.df_clustered = self.last_result["df_cleaned_filtered"]
                print("MLService: Loaded and processed default dataset successfully.")
                return
            except Exception as e:
                print(f"MLService: Warning - processing default dataset failed ({e}). Retraining...")
        
        self.artifacts, self.df_clustered = run_ml_pipeline()

    def is_healthy(self) -> bool:
        return self.artifacts is not None and self.df_clustered is not None and not self.df_clustered.empty

    def process_csv_file(self, file_bytes: bytes) -> dict:
        try:
            df_raw = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Invalid CSV file format: {str(e)}")
            
        result = process_dataset(df_raw)
        self.last_result = result
        self.artifacts = result["artifacts"]
        self.df_clustered = result["df_cleaned_filtered"]
        return self._format_result_response(result)

    def get_last_result_response(self) -> dict:
        if self.last_result is None:
            self.load_or_train()
        return self._format_result_response(self.last_result)

    def _format_result_response(self, result: dict) -> dict:
        df_raw_filt = result["df_raw_filtered"]
        df_cleaned_filt = result["df_cleaned_filtered"]
        cluster_summaries = result["cluster_summaries"]
        
        total_raw = result["total_raw"]
        total_processed = result["total_processed"]
        total_excluded = total_raw - total_processed
        
        records = []
        for idx, row in df_raw_filt.iterrows():
            rec = {
                "id": int(row.get("ID", idx + 1)),
                "year_birth": int(row.get("Year_Birth", 1980)),
                "education": str(row.get("Education", "Graduate")),
                "marital_status": str(row.get("Marital_Status", "Single")),
                "income": round(float(row.get("Income", 0)), 2) if not pd.isna(row.get("Income")) else 0,
                "recency": int(row.get("Recency", 0)),
                "total_spending": round(float(df_cleaned_filt.loc[idx, "Total_Spending"]), 2),
                "web_purchases": int(row.get("NumWebPurchases", 0)),
                "store_purchases": int(row.get("NumStorePurchases", 0)),
                "catalog_purchases": int(row.get("NumCatalogPurchases", 0)),
                "deals_purchases": int(row.get("NumDealsPurchases", 0)),
                "cluster": int(row.get("cluster", 0)),
                "pca1": float(row.get("PCA1", 0)),
                "pca2": float(row.get("PCA2", 0)),
                "pca3": float(row.get("PCA3", 0))
            }
            records.append(rec)

        pca_points = [
            {
                "x": float(row.get("PCA1", 0)),
                "y": float(row.get("PCA2", 0)),
                "cluster": int(row.get("cluster", 0)),
                "id": int(row.get("ID", idx + 1))
            }
            for idx, row in df_raw_filt.iterrows()
        ]

        counts = {f"Cluster {c['cluster_id']}": c["count"] for c in cluster_summaries}

        return {
            "total_raw": total_raw,
            "total_processed": total_processed,
            "total_excluded": total_excluded,
            "exclusion_reason": "Outliers excluded based on ML pipeline rules (Age >= 90 or Income >= $600,000)",
            "num_clusters": 4,
            "cluster_counts": counts,
            "cluster_summaries": cluster_summaries,
            "pca_points": pca_points,
            "customers": records
        }

    def generate_clustered_csv(self) -> bytes:
        if self.last_result is None or "df_raw_filtered" not in self.last_result:
            self.load_or_train()
        df_out = self.last_result["df_raw_filtered"].copy()
        
        buf = io.StringIO()
        df_out.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    def get_analytics(self) -> dict:
        if self.last_result is None:
            self.load_or_train()
        df = self.df_clustered
        total_customers = len(df)
        cluster_counts = df["cluster"].value_counts().to_dict()
        counts_formatted = {f"Cluster {k}": int(cluster_counts.get(k, 0)) for k in sorted(df["cluster"].unique())}
        
        return {
            "total_customers": total_customers,
            "num_clusters": self.artifacts.get("n_clusters", 4),
            "overall_avg_income": round(float(df["Income"].mean()), 2),
            "overall_avg_spending": round(float(df["Total_Spending"].mean()), 2),
            "overall_avg_recency": round(float(df["Recency"].mean()), 1),
            "overall_avg_age": round(float(df["Age"].mean()), 1),
            "cluster_distribution": counts_formatted,
            "pca_variance_ratio": [round(v, 4) for v in self.artifacts.get("pca_variance_ratio", [])]
        }

    def get_clusters_info(self) -> list[dict]:
        if self.last_result is None:
            self.load_or_train()
        return self.last_result["cluster_summaries"]

    def predict(self, raw_input: dict) -> dict:
        if self.artifacts is None:
            self.load_or_train()
        cluster_id, pca_coords, distances = preprocess_single_input(raw_input, self.artifacts)
        clusters_info = self.get_clusters_info()
        cluster_meta = next((c for c in clusters_info if c["cluster_id"] == cluster_id), None)
        
        return {
            "cluster": cluster_id,
            "cluster_name": cluster_meta["cluster_name"] if cluster_meta else f"Cluster {cluster_id}",
            "message": f"Customer assigned to Cluster {cluster_id}",
            "summary": cluster_meta if cluster_meta else {},
            "pca_coordinates": [round(x, 4) for x in pca_coords],
            "distances_to_centroids": [round(d, 4) for d in distances]
        }

ml_service = MLService()
