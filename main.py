from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import random
import numpy


def load_and_split_iris_data():
    ### Instructions [1. Data preparation] ###
    # Load all iris attributes (data) and species information (results)
    # There is no need to clear data, so attributes are loaded directly

    # X (data) attributes:
    # data[i][0] sepal length (cm)
    # data[i][1] sepal width (cm)
    # data[i][2] petal length (cm)
    # data[i][3] petal width (cm)

    # y[i] (target) = 0, 1, 2 (species of iris flower)
    # 0 = Iris Setosa
    # 1 = Iris Versicolor
    # 2 = Iris Virginica
    
    # Load the iris data from sklearn
    iris = load_iris()
    X, y = iris.data, iris.target

    ### Instructions [2. Data split] ###
    # X_train - training data
    # y_train - training data results
    # X_test - test data
    # y_test - test data results
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test), (X_train_scaled, X_test_scaled)

def kkjmah_manual_test():
    # "Inputs":
    doFullLog = False  # Logging all data (not only final report)

    (X_train, y_train), (X_test, y_test), (X_train_scaled, X_test_scaled) = load_and_split_iris_data()

    if(doFullLog):
        print(f"X_Train: {X_train}")
        print(f"y_train: {y_train}")
        print(f"X_Test: {X_test}")
        print(f"y_test: {y_test}")
        print(f"X_train_scaled: {X_train_scaled}")
        print(f"X_test_scaled: {X_test_scaled}")

    ### Instructions [3. Model definition] and [4. Model training] ###
    # Classifier preparation
    log_reg = LogisticRegression(max_iter=200)
    rf_clf = RandomForestClassifier(random_state=42)

    log_reg.fit(X_train_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_scaled)

    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    print("Feature names:", load_iris().feature_names)
    print("Feature importances:", rf_clf.feature_importances_)

    # Random guessing for reference
    y_pred_rng = numpy.array(random.choices([0, 1, 2], k=len(y_test)))

    print("\nPredictions:")
    print(f"{y_test} - True values")
    print(f"{y_pred_log} - Logistic Regression")
    print(f"{y_pred_rf} - Random Forest")
    print(f"{y_pred_rng} - Random Guessing")

    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log)*100, "%")
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf)*100, "%")
    print("Random Guessing Accuracy:", accuracy_score(y_test, y_pred_rng)*100, "%")

    print("\nClassification Report for Logistic Regression:\n", classification_report(y_test, y_pred_log))
    print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))
    print("\nClassification Report for Random Guessing:\n", classification_report(y_test, y_pred_rng))

# Writes results to a file
def kkjmah_automatic_test_log_reg():
    max_iters = [20, 50, 100, 200, 500]
    output_file = open("log_reg_results.txt", "w")
    (_, y_train), (_, y_test), (X_train_scaled, X_test_scaled) = load_and_split_iris_data()
    for max_iter in max_iters:
        log_reg = LogisticRegression(max_iter=max_iter)

        log_reg.fit(X_train_scaled, y_train)
        y_pred_log = log_reg.predict(X_test_scaled)

        output_file.write(f"{max_iter} {accuracy_score(y_test, y_pred_log)*100}%\n")
        # "Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log)*100, "%"
    output_file.close()

def kkjmah_automatic_test_rf_clf():
    numbers_of_estimators = [1, 2, 3, 5, 10, 50, 100, 200, 500]
    output_file = open("rf_clf_results.txt", "w")
    (X_train, y_train), (X_test, y_test), (_, _) = load_and_split_iris_data()
    for n_of_estimators in numbers_of_estimators:
        rf_clf = RandomForestClassifier(n_estimators=n_of_estimators, random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred_log = rf_clf.predict(X_test)

        output_file.write(f"{n_of_estimators} {accuracy_score(y_test, y_pred_log)*100}%\n")
    output_file.close()

if __name__ == "__main__":
    kkjmah_automatic_test_log_reg()
    kkjmah_automatic_test_rf_clf()
    kkjmah_manual_test()