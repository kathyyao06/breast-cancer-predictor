from scipy.stats import laplace
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

def cm(expected_overall, predicted_overall):
    #finding confusion matrix
    cm_overall = confusion_matrix(expected_overall, predicted_overall)
    TP = cm_overall[0][0]
    FP = cm_overall[0][1]
    FN = cm_overall[1][0]
    TN = cm_overall[1][1]
    print(f"Confusion matrix: \n1. True Positive: {TP}\n2. False Positive: {FP}\n3. True Negative: {TN}\n4. False Negative {FN}")
    
    #calculating FPR and FNR
    overall_FPR = FP / (FP + TN)
    overall_FNR = FN / (FN + TP)
    print(f"\nFalse Positive Rate: {overall_FPR}\nFalse Negative Rate: {overall_FNR}\n")
 
def decision_tree(X_train, X_test, y_train, y_test):
    # Train Decision Tree Classifer
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    #Data about model performance
    print("Decision Tree Model")
    print(classification_report(y_test, y_pred))
    cm(y_test, y_pred)


def logistic_reg(X_train, X_test, y_train, y_test):
    #Using logistic regression model
    logmodel = LogisticRegression(max_iter=3000)
    logmodel.fit(X_train, y_train)

    #Predicting Response
    y_pred = logmodel.predict(X_test)

    #Data about model performance
    print("Logistic Regression Model")
    print(classification_report(y_test,y_pred))
    cm(y_test, y_pred)


def svm_model(X_train, X_test, y_train, y_test):
    #Training Classifier
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    linear_model = svm.SVC(kernel = 'linear')
    linear_model.fit(X_train, y_train)

    #Predicting Response
    y_pred = linear_model.predict(X_test)

    #Data about model performance
    print("SVM Model")
    print(classification_report(y_test,y_pred))
    cm(y_test, y_pred)

def main():
    df = pd.read_csv('data.csv')
    
    #Data cleaning and Data preparation
    diagnosis = pd.get_dummies(df['diagnosis'], drop_first = True)
    df = pd.concat([diagnosis, df], axis = 1)
    df = df.drop(["id", "diagnosis"], axis = 1)
    df.fillna(0, inplace = True)

    #Model Training
    data = df.drop(['M'], axis = 1)
    target = df['M']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.3)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    #Creating three types of models
    svm_model(X_train, X_test, y_train, y_test)
    logistic_reg(X_train, X_test, y_train, y_test)
    decision_tree(X_train, X_test, y_train, y_test)

if __name__=="__main__":
    main()
