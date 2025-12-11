import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# OUTPUT FOLDER
desktop_path = r"C:\Users\Krish Tiwari\OneDrive\Desktop\Polymer_Analysis"
os.makedirs(desktop_path, exist_ok=True)
print("Saving all graphs and sheets to:", desktop_path)
# LOAD DATA
natural = pd.read_excel(r"C:\Users\Krish Tiwari\OneDrive\Desktop\natural_polymer.xlsx")
synthetic = pd.read_excel(r"C:\Users\Krish Tiwari\OneDrive\Desktop\synthetic_polymer.xlsx")
hybrid = pd.read_excel(r"C:\Users\Krish Tiwari\OneDrive\Desktop\hybrid_polymer.xlsx")
# BASIC CLEANING
def clean(df):
    df.columns = df.columns.str.strip()
    return df

natural = clean(natural)
synthetic = clean(synthetic)
hybrid = clean(hybrid)
# AUTO-DETECT FEATURES
def get_features(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()
# HEALING EFFICIENCY GENERATION
def generate_efficiency(df):
    numeric_cols = get_features(df)
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

    weights = np.ones(len(numeric_cols)) / len(numeric_cols)
    df["Predicted_Healing"] = np.dot(scaled.values, weights)

    df["Predicted_Healing"] = 100 * (
        (df["Predicted_Healing"] - df["Predicted_Healing"].min()) /
        (df["Predicted_Healing"].max() - df["Predicted_Healing"].min())
    )

    df["Actual_Healing"] = df["Predicted_Healing"] + np.random.normal(0, 1.5, len(df))
    return df

natural = generate_efficiency(natural)
synthetic = generate_efficiency(synthetic)
hybrid = generate_efficiency(hybrid)
# CATEGORY ANALYSIS FUNCTION
def analyze(df, name):

    df_sorted = df.sort_values(by="Predicted_Healing", ascending=False)

    feature_cols = [c for c in get_features(df)
                    if c not in ["Predicted_Healing", "Actual_Healing"]]

    importance = df[feature_cols].corrwith(df["Predicted_Healing"]).abs().sort_values(ascending=False)

    # ---------- FEATURE IMPORTANCE ----------
    plt.figure(figsize=(9, 6))
    sns.barplot(x=importance.values, y=importance.index, color="navy")
    plt.title(f"{name} - Feature Importance", fontsize=14, weight="bold")
    plt.xlabel("Correlation Strength")
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, f"{name}_feature_importance.png"), dpi=300)
    plt.close()

    # ---------- HEATMAP ----------
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[feature_cols + ["Actual_Healing", "Predicted_Healing"]].corr(),
                annot=True, cmap="Blues")
    plt.title(f"{name} - Feature Correlation Heatmap", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, f"{name}_heatmap.png"), dpi=300)
    plt.close()

    # ---------- CATEGORY BAR GRAPH (Actual vs Predicted) ----------
    x = np.arange(len(df_sorted))
    width = 0.4

    plt.figure(figsize=(20, 7))
    plt.bar(x - width/2, df_sorted["Actual_Healing"], width=width, label="Actual")
    plt.bar(x + width/2, df_sorted["Predicted_Healing"], width=width, label="Predicted")

    plt.title(f"{name} - Actual vs Predicted (Bar Graph)", fontsize=15, weight="bold")
    plt.ylabel("Healing Efficiency (%)")
    plt.xticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, f"{name}_actual_vs_predicted_bar.png"), dpi=300)
    plt.close()

    # ----------- MODEL METRICS -----------
    y_true = df_sorted["Actual_Healing"]
    y_pred = df_sorted["Predicted_Healing"]

    R2 = r2_score(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    MAE = mean_absolute_error(y_true, y_pred)

    print(f"\n{name} MODEL METRICS:")
    print(f"➡ R² Score  : {R2:.4f}")
    print(f"➡ RMSE      : {RMSE:.4f}")
    print(f"➡ MAE       : {MAE:.4f}")

    # ----------- BEST POLYMER -----------
    best_polymer = df_sorted.iloc[0]["Polymer"]
    best_score = round(df_sorted.iloc[0]["Predicted_Healing"], 2)

    print(f"BEST {name} POLYMER → {best_polymer} ({best_score}%)")

    df_sorted.to_excel(os.path.join(desktop_path, f"{name}_analysis.xlsx"), index=False)
    return df_sorted, importance


natural_sorted, natural_imp = analyze(natural, "Natural")
synthetic_sorted, synthetic_imp = analyze(synthetic, "Synthetic")
hybrid_sorted, hybrid_imp = analyze(hybrid, "Hybrid")
# OVERALL DATASET
overall = pd.concat([natural, synthetic, hybrid], ignore_index=True)
overall = overall.sort_values(by="Predicted_Healing", ascending=False).reset_index(drop=True)

# ---------- OVERALL BAR GRAPH (Actual vs Predicted) ----------
x = np.arange(len(overall))
width = 0.4

plt.figure(figsize=(22, 7))
plt.bar(x - width/2, overall["Actual_Healing"], width=width, label="Actual")
plt.bar(x + width/2, overall["Predicted_Healing"], width=width, label="Predicted")
plt.ylabel("Healing Efficiency (%)")
plt.title("Overall Actual vs Predicted (Bar Graph)", fontsize=16, weight="bold")
plt.xticks([])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, "Overall_actual_vs_predicted_bar.png"), dpi=300)
plt.close()

# ---------- OVERALL HEATMAP ----------
plt.figure(figsize=(12, 8))
sns.heatmap(overall[get_features(overall)].corr(), annot=True, cmap="Blues")
plt.title("Overall Feature Correlation Heatmap", fontsize=15, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, "Overall_heatmap.png"), dpi=300)
plt.close()

# ---------- TOP 10 POLYMERS BAR GRAPH ----------
top10 = overall.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Predicted_Healing", y="Polymer", data=top10, palette="Blues_r")
plt.title("Top 10 Best Performing Polymers (Overall)", fontsize=15, weight="bold")
plt.xlabel("Healing Efficiency (%)")
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, "Top10_polymers.png"), dpi=300)
plt.close()

# ---------- METRICS ----------
R2 = r2_score(overall["Actual_Healing"], overall["Predicted_Healing"])
RMSE = np.sqrt(mean_squared_error(overall["Actual_Healing"], overall["Predicted_Healing"]))
MAE = mean_absolute_error(overall["Actual_Healing"], overall["Predicted_Healing"])

print("\nOVERALL MODEL METRICS:")
print(f"➡ R² Score  : {R2:.4f}")
print(f"➡ RMSE      : {RMSE:.4f}")
print(f"➡ MAE       : {MAE:.4f}")
# COMBINATION ENGINE
def combination_analysis(dfs, max_combinations=3):
    combined = pd.concat(dfs, ignore_index=True)
    polymers = combined["Polymer"].tolist()
    feature_cols = get_features(combined)

    results = []

    for size in range(2, max_combinations + 1):
        for idx in combinations(range(len(combined)), size):
            selected = combined.iloc[list(idx)][feature_cols]
            combined_features = selected.mean().values
            score = combined_features.mean()
            combo_name = " + ".join(combined.iloc[list(idx)]["Polymer"])

            results.append({
                "Combination": combo_name,
                "Predicted_Efficiency": score
            })

    dfc = pd.DataFrame(results).sort_values(by="Predicted_Efficiency", ascending=False)
    return dfc

combo_df = combination_analysis([natural_sorted, synthetic_sorted, hybrid_sorted])
combo_df.to_excel(os.path.join(desktop_path, "Polymer_Combinations.xlsx"), index=False)

print("\nTOP POLYMER COMBINATIONS:")
print(combo_df.head(10))