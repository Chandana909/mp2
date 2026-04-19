import os
import joblib
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add parent directory to path so we can import the client we just made
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_platform_client import MLPlatformClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def build_dummy_model(output_path: str):
    """Build a simple Random Forest model and save it to a joblib file."""
    logging.info("🧠 Training a new sample Random Forest model...")
    # 5 features to match exactly what we will predict later
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, output_path)
    logging.info(f"💾 Saved sample model to {output_path}")

def main():
    print("=" * 60)
    print("🚀 ML PLATFORM DIRECT CLIENT DEMO")
    print("=" * 60)
    
    # 1. Initialize the Direct Client
    client = MLPlatformClient("http://127.0.0.1:8000")
    
    model_name = "direct_demo_model"
    model_path = "sample_direct_model.joblib"
    features = ["f1", "f2", "f3", "f4", "f5"]
    
    try:
        # 2. Build and save a model locally
        build_dummy_model(model_path)
        
        # --- UPLOAD NEW MODEL directly ---
        print("\n--- 1. UPLOADING NEW MODEL ---")
        upload_resp = client.upload_model(
            file_path=model_path,
            model_name=model_name,
            version="v1.0",
            features=features
        )
        print(f"Server Response: {upload_resp}")
        time.sleep(1) # small pause for realism
        
        # NOTE: After upload, the model is in 'candidate' state. 
        # In a real environment you would call /promote to make it active, but 
        # the retrain pipeline handles finding models as well. 
        # For predict to work, let's manually hit the promote endpoint to ensure it's active.
        print("\n--- 2. PROMOTING MODEL TO ACTIVE ---")
        logging.info("Promoting model to make it active right away...")
        promote_resp = client.client.post(f"{client.base_url}/models/{model_name}/promote")
        print(f"Promote Response: {promote_resp.json()}")
        time.sleep(1)
        
        # --- GET PIEPLINE OUTPUTS directly ---
        print("\n--- 3. TRIGGERING RETRAINING PIPELINE ---")
        pipeline_resp = client.trigger_pipeline(model_name)
        new_version = pipeline_resp.get("new_version")
        new_acc = pipeline_resp.get("new_accuracy")
        print(f"Pipeline finished! Generated new version '{new_version}' with Accuracy {new_acc}")
        time.sleep(1)
        
        # --- GET PREDICTIONS directly ---
        print("\n--- 4. TAKING DIRECT PREDICTION ---")
        sample_features = {"f1": 0.5, "f2": -1.2, "f3": 3.4, "f4": 0.0, "f5": -0.8}
        
        prediction_resp = client.predict(
            model_name=model_name,
            features=sample_features
        )
        print("Final Direct Prediction Output:")
        print(f"  -> Predicted Class: {prediction_resp.get('prediction')}")
        print(f"  -> Confidence: {prediction_resp.get('confidence'):.2%}")
        
    except FileNotFoundError:
        logging.error(f"Could not find model file {model_path}.")
    except Exception as e:
        logging.error(f"Error during usage: {e}")
    finally:
        # Cleanup the dummy file
        if os.path.exists(model_path):
            os.remove(model_path)
            logging.info("🧹 Cleaned up local dummy joblib file.")
            
    print("\n✅ DEMO COMPLETE")

if __name__ == "__main__":
    main()
