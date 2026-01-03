# src/tree_utils.py
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import graphviz

def train_and_evaluate(X_train, y_train, X_test, y_test, max_depth=None):
    # Train model [cite: 36, 37]
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Report & Matrix 
    print(f"--- Max Depth: {max_depth} ---")
    print(classification_report(y_test, y_pred))
    
    # Return accuracy for the table requirement [cite: 105]
    acc = accuracy_score(y_test, y_pred)
    return clf, acc

def visualize_tree(clf, feature_names, class_names):
    # Wrapper để vẽ cây với Graphviz [cite: 37, 104]
    dot_data = export_graphviz(clf, out_file=None, 
                               feature_names=feature_names,  
                               class_names=class_names,  
                               filled=True, rounded=True)  
    return graphviz.Source(dot_data)