
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_DIR, "abalone"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "breast_cancer"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "diabetes"), exist_ok=True)

def process_breast_cancer():
    print("--- Processing Breast Cancer ---")
    data_path = os.path.join(DATA_DIR, "custom_dataset", "breast_cancer.csv")
    out_dir = os.path.join(OUTPUT_DIR, "breast_cancer")
    
    if not os.path.exists(data_path):
        print(f"Error: File not found {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=['ID', 'Diagnosis'])
    y = df['Diagnosis']
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
    )
    
    feature_names = X.columns.tolist()
    class_names = [str(c) for c in sorted(y.unique())]
    depths = [2, 3, 4, 5, 6, 7, None]
    results = []
    
    for depth in depths:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        depth_label = str(depth) if depth is not None else "None"
        results.append((depth_label, acc))
        
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, fontsize=10)
        plt.title(f"Decision Tree (Max Depth = {depth_label})")
        plt.savefig(os.path.join(out_dir, f"Cay_Depth_{depth_label}.png"), bbox_inches='tight')
        plt.close()
        print(f"Saved Breast Cancer Tree Depth {depth_label}")

    # Accuracy Chart
    df_res = pd.DataFrame(results, columns=["Max Depth", "Accuracy"])
    plt.figure(figsize=(10, 5))
    plt.plot(df_res["Max Depth"], df_res["Accuracy"], marker='o', color='purple', linestyle='-')
    plt.title("Breast Cancer: Accuracy vs Depth")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(out_dir, "Accuracy_Chart.png"))
    plt.close()
    print("Saved Breast Cancer Accuracy Chart")

def process_diabetes():
    print("--- Processing Diabetes ---")
    data_path = os.path.join(DATA_DIR, "diabetes", "diabetes.csv")
    out_dir = os.path.join(OUTPUT_DIR, "diabetes")
    
    if not os.path.exists(data_path):
        print(f"Error: File not found {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Label Encoding
    target_col = 'class'
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == object:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
            
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Ratios as per notebook
    ratios = [0.4, 0.6, 0.8, 0.9]
    
    for train_size in ratios:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=42
        )
        
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        clf.fit(X_train, y_train)
        
        # Save Tree
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=X.columns.tolist(), class_names=['Negative', 'Positive'], filled=True, fontsize=10)
        split_name = f"{int(train_size*100)}_{int((1-train_size)*100)}"
        plt.title(f"Diabetes Decision Tree (Split {split_name})")
        plt.savefig(os.path.join(out_dir, f"Tree_Split_{split_name}.png"), bbox_inches='tight')
        plt.close()
        print(f"Saved Diabetes Tree Split {split_name}")

def process_abalone():
    print("--- Processing Abalone ---")
    data_path = os.path.join(DATA_DIR, "abalone", "abalone.csv")
    out_dir = os.path.join(OUTPUT_DIR, "abalone")
    
    if not os.path.exists(data_path):
        print(f"Error: File not found {data_path}")
        return

    col_names = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
                 "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
    # Assuming file has no header based on notebook usage
    df = pd.read_csv(data_path, header=None, names=col_names)
    
    # Preprocessing
    df = pd.get_dummies(df, columns=['Sex'], drop_first=False)
    
    class_counts = df['Rings'].value_counts()
    valid_rings = class_counts[class_counts >= 10].index
    df_clean = df[df['Rings'].isin(valid_rings)].copy()
    
    X = df_clean.drop('Rings', axis=1)
    y = df_clean['Rings']
    
    # Train 80/20 split for demo visualization
    train_size = 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=42
    )
    
    clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)
    
    # Save Tree (can be very large, limit depth for viz)
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns.tolist(), class_names=[str(c) for c in sorted(y.unique())], filled=True, max_depth=3, fontsize=10)
    plt.title("Abalone Decision Tree (Split 80/20, Max Depth 3)")
    plt.savefig(os.path.join(out_dir, "Abalone_Tree_80_20_Depth3.png"), bbox_inches='tight')
    plt.close()
    
    # Save Full Tree PDF/PNG if possible (might be too big)
    # Using export_graphviz for higher quality if needed, but matplotlib is safer to run here.
    print("Saved Abalone Tree")

if __name__ == "__main__":
    process_breast_cancer()
    process_diabetes()
    process_abalone()
