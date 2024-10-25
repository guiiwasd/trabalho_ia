import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Carrega o dataset CIFAR-10
cifar10 = fetch_openml('cifar_10', version=1, cache=True)

# Separa os dados e os rótulos
X = cifar10.data
y = cifar10.target

# Converte os rótulos para inteiros
y = y.astype(np.uint8)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# 1. Classificador SGD
sgdClf = SGDClassifier(random_state=42)
sgdClf.fit(X_train, y_train)
y_pred_sgd = sgdClf.predict(X_test)

print("SGD Classifier:")
print(classification_report(y_test, y_pred_sgd))
print(f"Accuracy: {accuracy_score(y_test, y_pred_sgd)}\n")

# 2. Classificador RandomForest
rfClf = RandomForestClassifier(n_estimators=100, random_state=42)
rfClf.fit(X_train, y_train)
y_pred_rf = rfClf.predict(X_test)

print("Random Forest Classifier:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}\n")

# 3. Classificador SVM
svmClf = svm.SVC(kernel='linear')
svmClf.fit(X_train, y_train)
y_pred_svm = svmClf.predict(X_test)

print("SVM Classifier:")
print(classification_report(y_test, y_pred_svm))
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}\n")

# Comparando métricas em porcentagem
metrics = {
    "SGD": {
        "Accuracy": accuracy_score(y_test, y_pred_sgd) * 100,
        "Recall": recall_score(y_test, y_pred_sgd, average="weighted") * 100,
        "Precision": precision_score(y_test, y_pred_sgd, average="weighted") * 100,
        "F1-Score": f1_score(y_test, y_pred_sgd, average="weighted") * 100
    },
    "RandomForest": {
        "Accuracy": accuracy_score(y_test, y_pred_rf) * 100,
        "Recall": recall_score(y_test, y_pred_rf, average="weighted") * 100,
        "Precision": precision_score(y_test, y_pred_rf, average="weighted") * 100,
        "F1-Score": f1_score(y_test, y_pred_rf, average="weighted") * 100
    },
    "SVM": {
        "Accuracy": accuracy_score(y_test, y_pred_svm) * 100,
        "Recall": recall_score(y_test, y_pred_svm, average="weighted") * 100,
        "Precision": precision_score(y_test, y_pred_svm, average="weighted") * 100,
        "F1-Score": f1_score(y_test, y_pred_svm, average="weighted") * 100
    }    
}

print("Comparação de Métricas:")
for clf, clf_metrics in metrics.items():
    print(f"\nClassificador: {clf}")
    for metric, score in clf_metrics.items():
        print(f"{metric}: {score:.2f}%")