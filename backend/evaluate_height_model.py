import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# Ensure the backend directory is in the path to import services
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from services.height_estimator import estimate_height

def evaluate_and_calibrate():
    dataset_dir = Path(r"d:\pre-capstone-main\dataset_of_height_weight_updated")
    csv_path = dataset_dir / "Output_data.csv"
    images_dir = dataset_dir / "images"

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    actual_heights = []
    estimated_heights = []
    valid_indices = []
    
    print("\n--- Running Inference ---")
    print(f"{'Image Name':<40} | {'Actual (cm)':<12} | {'Estimated (cm)':<15}")
    print("-" * 75)

    for idx, row in df.iterrows():
        image_name = row["Image Name"]
        actual_height = row["Height (cm)"]
        dist_cm = row["Distance from camera (cm)"]
        cam_height_cm = row["Camera height from ground (cm)"]
        
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found {image_path}")
            continue

        try:
            result = estimate_height(
                image_path=str(image_path),
                camera_height_cm=cam_height_cm,
                distance_cm=dist_cm
            )
            est_height = result["estimated_height_cm"]
            
            actual_heights.append(actual_height)
            estimated_heights.append(est_height)
            valid_indices.append(idx)
            
            print(f"{image_name:<40} | {actual_height:<12.1f} | {est_height:<15.1f}")
            
        except Exception as e:
            print(f"{image_name:<40} | {actual_height:<12.1f} | ERROR: {e}")

    if not actual_heights:
        print("No valid inferences were made.")
        return

    actual_heights = np.array(actual_heights)
    estimated_heights = np.array(estimated_heights)

    # 1. Evaluate Current Model
    mae_before = mean_absolute_error(actual_heights, estimated_heights)
    rmse_before = np.sqrt(mean_squared_error(actual_heights, estimated_heights))

    print("\n" + "="*50)
    print(" BASE MODEL PERFORMANCE (Geometric Pinhole)")
    print("="*50)
    print(f"Mean Absolute Error (MAE) : {mae_before:.2f} cm")
    print(f"Root Mean Squared Error   : {rmse_before:.2f} cm")

    # 2. Train Calibration Model
    print("\n--- Training Calibration Model (Linear Regression) ---")
    X = estimated_heights.reshape(-1, 1)
    y = actual_heights

    reg = LinearRegression()
    reg.fit(X, y)

    # The equation is: Actual_Height = m * Estimated_Height + c
    m = reg.coef_[0]
    c = reg.intercept_
    print(f"Calibration Equation: Actual_Height = {m:.4f} * Estimated_Height + {c:.4f}")

    # 3. Evaluate Calibrated Model
    calibrated_heights = reg.predict(X)
    
    mae_after = mean_absolute_error(actual_heights, calibrated_heights)
    rmse_after = np.sqrt(mean_squared_error(actual_heights, calibrated_heights))

    print("\n" + "="*50)
    print(" CALIBRATED MODEL PERFORMANCE")
    print("="*50)
    print(f"Mean Absolute Error (MAE) : {mae_after:.2f} cm")
    print(f"Root Mean Squared Error   : {rmse_after:.2f} cm")
    
    improvement_mae = mae_before - mae_after
    print(f"\nImprovement in MAE: {improvement_mae:.2f} cm")

    # --- Generate Graphs ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # 1. Actual vs Estimated Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_heights, estimated_heights, color='red', alpha=0.6, label='Base Model (Uncalibrated)')
    plt.scatter(actual_heights, calibrated_heights, color='green', alpha=0.6, label='Calibrated Model')
    # Ideal line
    min_val = min(actual_heights.min(), estimated_heights.min(), calibrated_heights.min()) - 5
    max_val = max(actual_heights.max(), estimated_heights.max(), calibrated_heights.max()) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (Perfect Accuracy)')
    plt.xlabel('Actual Height (cm)')
    plt.ylabel('Predicted Height (cm)')
    plt.title('Actual vs Predicted Height')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataset_dir / "accuracy_scatter.png")
    plt.close()

    # 2. Error Distribution Histogram
    errors_before = estimated_heights - actual_heights
    errors_after = calibrated_heights - actual_heights

    plt.figure(figsize=(10, 6))
    sns.histplot(errors_before, color='red', kde=True, label=f'Base Error (MAE: {mae_before:.1f}cm)', alpha=0.4, bins=15)
    sns.histplot(errors_after, color='green', kde=True, label=f'Calibrated Error (MAE: {mae_after:.1f}cm)', alpha=0.6, bins=15)
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Error (Predicted - Actual) in cm')
    plt.ylabel('Count of Images')
    plt.title('Error Distribution (Before vs After Calibration)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dataset_dir / "error_distribution.png")
    plt.close()

    # 3. MAE Improvement Bar Chart
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Base Model\n(Geometric)', 'Calibrated Model\n(Machine Learning)'], [mae_before, mae_after], color=['red', 'green'])
    plt.ylabel('Mean Absolute Error (cm)')
    plt.title('Model Accuracy Improvement')
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} cm', ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, max(mae_before, mae_after) + 5)
    plt.tight_layout()
    plt.savefig(dataset_dir / "mae_improvement.png")
    plt.close()

    print(f"\nGraphs saved to: {dataset_dir}")

if __name__ == "__main__":
    evaluate_and_calibrate()
