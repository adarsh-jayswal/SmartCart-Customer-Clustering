import os
import io
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "smartcart_customers.csv"))
ARTIFACTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "artifacts"))

REQUIRED_COLUMNS = [
    "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Dt_Customer", "Recency", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"
]

def validate_dataset_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {', '.join(missing)}")

def process_dataset(df_raw: pd.DataFrame):
    """
    Executes the exact preprocessing, feature engineering, encoding, scaling, PCA,
    and Agglomerative Clustering pipeline from smartcart.ipynb on a given dataframe.
    """
    validate_dataset_columns(df_raw)
    
    df = df_raw.copy()
    
    # 1. Missing income handling
    income_median = float(df["Income"].median()) if not df["Income"].isnull().all() else 50000.0
    df["Income"] = df["Income"].fillna(income_median)
    
    # 2. Feature engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer_Parsed"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
    reference_date = df["Dt_Customer_Parsed"].max()
    if pd.isna(reference_date):
        reference_date = pd.to_datetime("2014-06-29")
        
    df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer_Parsed"]).dt.days.fillna(1000)
    
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    
    # Map Education & Marital Status
    df["Education_Mapped"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    })
    
    df["Living_With_Mapped"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    })
    
    # Drop raw columns unused in ML model
    cols_to_drop = [
        "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer", "Dt_Customer_Parsed",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df_cleaned = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df_cleaned["Education"] = df["Education_Mapped"]
    df_cleaned["Living_With"] = df["Living_With_Mapped"]
    df_cleaned.drop(columns=["Education_Mapped", "Living_With_Mapped"], inplace=True, errors="ignore")
    
    # Outlier removal
    outlier_mask = (df_cleaned["Age"] < 90) & (df_cleaned["Income"] < 600000)
    df_cleaned_filtered = df_cleaned[outlier_mask].reset_index(drop=True)
    df_raw_filtered = df.iloc[df[outlier_mask].index].reset_index(drop=True)
    
    # One-Hot Encoding
    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder(sparse_output=False)
    enc_cols = ohe.fit_transform(df_cleaned_filtered[cat_cols])
    enc_df = pd.DataFrame(
        enc_cols,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df_cleaned_filtered.index
    )
    
    df_encoded = pd.concat([df_cleaned_filtered.drop(columns=cat_cols), enc_df], axis=1)
    feature_names = list(df_encoded.columns)
    
    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    # PCA (3 components)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Agglomerative Clustering (K=4, Ward linkage)
    agg_clf = AgglomerativeClustering(n_clusters=4, linkage="ward")
    cluster_labels = agg_clf.fit_predict(X_pca)
    
    # Attach clusters and PCA coordinates
    df_raw_filtered["cluster"] = cluster_labels
    df_raw_filtered["PCA1"] = np.round(X_pca[:, 0], 4)
    df_raw_filtered["PCA2"] = np.round(X_pca[:, 1], 4)
    df_raw_filtered["PCA3"] = np.round(X_pca[:, 2], 4)
    
    df_cleaned_filtered["cluster"] = cluster_labels
    df_cleaned_filtered["PCA1"] = np.round(X_pca[:, 0], 4)
    df_cleaned_filtered["PCA2"] = np.round(X_pca[:, 1], 4)
    df_cleaned_filtered["PCA3"] = np.round(X_pca[:, 2], 4)
    
    # Calculate Cluster Summary Stats
    total_len = len(df_cleaned_filtered)
    cluster_summaries = []
    
    for c in sorted(np.unique(cluster_labels)):
        sub = df_cleaned_filtered[df_cleaned_filtered["cluster"] == c]
        sub_raw = df_raw_filtered[df_raw_filtered["cluster"] == c]
        c_len = len(sub)
        pct = round((c_len / total_len) * 100, 1)
        
        avg_income = float(sub["Income"].mean())
        avg_spending = float(sub["Total_Spending"].mean())
        avg_recency = float(sub["Recency"].mean())
        avg_age = float(sub["Age"].mean())
        avg_tenure = float(sub["Customer_Tenure_Days"].mean())
        avg_children = float(sub["Total_Children"].mean())
        top_edu = str(sub["Education"].mode()[0]) if not sub["Education"].empty else "Graduate"
        top_living = str(sub["Living_With"].mode()[0]) if not sub["Living_With"].empty else "Partner"
        
        # Characteristic labels
        if avg_spending > 800 and top_living == "Partner":
            cluster_name = f"Cluster {c}: High-Value Partnered Buyers"
        elif avg_spending > 800 and top_living == "Alone":
            cluster_name = f"Cluster {c}: High-Value Single Buyers"
        elif avg_spending <= 800 and top_living == "Partner":
            cluster_name = f"Cluster {c}: Budget Partnered Household"
        else:
            cluster_name = f"Cluster {c}: Budget Single Household"

        spending_level = "high purchasing activity" if avg_spending > 800 else "moderate/budget spending"
        income_level = "high income" if avg_income > 60000 else "moderate/low income"
        
        interpretation = (
            f"Cluster {c} ({pct}% of dataset, {c_len} customers) represents {income_level} customers "
            f"with {spending_level} (avg ${avg_spending:,.0f} total spending). Members typically belong to a "
            f"{top_living.lower()} household with an average of {avg_children:.1f} children."
        )

        cluster_summaries.append({
            "cluster_id": int(c),
            "cluster_name": cluster_name,
            "count": c_len,
            "percentage": pct,
            "avg_income": round(avg_income, 2),
            "avg_spending": round(avg_spending, 2),
            "avg_recency": round(avg_recency, 1),
            "avg_age": round(avg_age, 1),
            "avg_tenure_days": round(avg_tenure, 0),
            "avg_children": round(avg_children, 2),
            "dominant_education": top_edu,
            "dominant_living_with": top_living,
            "interpretation": interpretation
        })
        
    centroids = np.array([X_pca[cluster_labels == k].mean(axis=0) for k in range(4)])
    
    artifacts = {
        "ohe": ohe,
        "scaler": scaler,
        "pca": pca,
        "centroids": centroids,
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "income_median": income_median,
        "reference_date": reference_date,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_clusters": 4
    }

    return {
        "df_raw_filtered": df_raw_filtered,
        "df_cleaned_filtered": df_cleaned_filtered,
        "cluster_summaries": cluster_summaries,
        "artifacts": artifacts,
        "total_processed": total_len,
        "total_raw": len(df_raw)
    }

def run_ml_pipeline():
    """Default pipeline runner for smartcart_customers.csv"""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df_raw = pd.read_csv(DATA_PATH)
    res = process_dataset(df_raw)
    
    artifacts_path = os.path.join(ARTIFACTS_DIR, "model_artifacts.pkl")
    with open(artifacts_path, "wb") as f:
        pickle.dump(res["artifacts"], f)
        
    dataset_out_path = os.path.join(ARTIFACTS_DIR, "clustered_dataset.csv")
    res["df_cleaned_filtered"].to_csv(dataset_out_path, index=False)
    
    print(f"Pipeline executed successfully. Artifacts saved to {ARTIFACTS_DIR}")
    return res["artifacts"], res["df_cleaned_filtered"]

def preprocess_single_input(raw_input: dict, artifacts: dict):
    ohe = artifacts["ohe"]
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    centroids = artifacts["centroids"]
    feature_names = artifacts["feature_names"]
    income_median = artifacts["income_median"]
    
    income = raw_input.get("Income")
    if income is None or income == "" or pd.isna(income):
        income = income_median
    else:
        income = float(income)
        
    year_birth = int(raw_input.get("Year_Birth", 1980))
    age = 2026 - year_birth
    tenure_days = int(raw_input.get("Customer_Tenure_Days", 1000))
    
    mnt_wines = float(raw_input.get("MntWines", 0))
    mnt_fruits = float(raw_input.get("MntFruits", 0))
    mnt_meat = float(raw_input.get("MntMeatProducts", 0))
    mnt_fish = float(raw_input.get("MntFishProducts", 0))
    mnt_sweet = float(raw_input.get("MntSweetProducts", 0))
    mnt_gold = float(raw_input.get("MntGoldProds", 0))
    
    total_spending = raw_input.get("Total_Spending")
    if total_spending is None:
        total_spending = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    else:
        total_spending = float(total_spending)
        
    kidhome = int(raw_input.get("Kidhome", 0))
    teenhome = int(raw_input.get("Teenhome", 0))
    total_children = raw_input.get("Total_Children")
    if total_children is None:
        total_children = kidhome + teenhome
    else:
        total_children = int(total_children)
        
    education_raw = str(raw_input.get("Education", "Graduation"))
    education_mapped = {
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate"
    }.get(education_raw, education_raw)
    if education_mapped not in ["Graduate", "Postgraduate", "Undergraduate"]:
        education_mapped = "Graduate"
        
    marital_raw = str(raw_input.get("Marital_Status", "Single"))
    living_with_mapped = {
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
    }.get(marital_raw, marital_raw)
    if living_with_mapped not in ["Alone", "Partner"]:
        living_with_mapped = "Alone"
        
    num_data = {
        "Income": income,
        "Recency": float(raw_input.get("Recency", 30)),
        "NumDealsPurchases": int(raw_input.get("NumDealsPurchases", 1)),
        "NumWebPurchases": int(raw_input.get("NumWebPurchases", 2)),
        "NumCatalogPurchases": int(raw_input.get("NumCatalogPurchases", 1)),
        "NumStorePurchases": int(raw_input.get("NumStorePurchases", 3)),
        "NumWebVisitsMonth": int(raw_input.get("NumWebVisitsMonth", 5)),
        "Complain": int(raw_input.get("Complain", 0)),
        "Response": int(raw_input.get("Response", 0)),
        "Age": age,
        "Customer_Tenure_Days": tenure_days,
        "Total_Spending": total_spending,
        "Total_Children": total_children
    }
    
    cat_df = pd.DataFrame([{
        "Education": education_mapped,
        "Living_With": living_with_mapped
    }])
    
    enc_array = ohe.transform(cat_df)
    enc_df = pd.DataFrame(enc_array, columns=ohe.get_feature_names_out(["Education", "Living_With"]))
    
    row_df = pd.DataFrame([num_data])
    full_row = pd.concat([row_df, enc_df], axis=1)[feature_names]
    
    scaled_row = scaler.transform(full_row)
    pca_row = pca.transform(scaled_row)
    
    distances = np.linalg.norm(pca_row - centroids, axis=1)
    predicted_cluster = int(np.argmin(distances))
    
    return predicted_cluster, pca_row[0].tolist(), distances.tolist()

if __name__ == "__main__":
    run_ml_pipeline()
