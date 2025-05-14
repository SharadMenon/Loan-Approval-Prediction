import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Loan_approval_sample_dataset.csv")
df = df.iloc[:,1:] #Unwanted column
df.info()
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Dependents'].unique()
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True) #3+ is not numerical so we consider it as categorical column
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].median(), inplace=True)

X= df.drop('Credit_History', axis=1)
y = df['Credit_History']
#Encoding the categorical variables
cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

column_transformer = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(), cols)],  
    remainder='passthrough'
)
X = column_transformer.fit_transform(X)

#Splititng into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#Building the model
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred1 = classifier.predict(X_test)
print("For Decision Tree: ")
print("Decision Tree Accuracy : ", accuracy_score(y_test, y_pred1))
print(classification_report(y_test,y_pred1))
print(confusion_matrix(y_test, y_pred1))
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier()
classifier2.fit(X_train,y_train)
y_pred2 = classifier.predict(X_test)

print("For Random Forest: ")
print("Random Forest Accuracy : ", accuracy_score(y_test, y_pred2))
print(classification_report(y_test,y_pred2))
print(confusion_matrix(y_test, y_pred2))

#Lets do visualization for some numerical factors and credit score
X_vis = df[['ApplicantIncome', 'LoanAmount']]
y_vis = df['Credit_History']

X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(X_vis,y_vis,test_size=0.2)

classifier3 = DecisionTreeClassifier()
classifier3.fit(X_vis_train,y_vis_train)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_vis_train.values, y_vis_train.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 100),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 10))
plt.contourf(X1, X2, classifier3.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_vis_test.values, y_vis_test.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 100),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 10))
plt.contourf(X1, X2, classifier3.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.legend()
plt.show()

classifier4 = RandomForestClassifier()
classifier4.fit(X_vis_train,y_vis_train)
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_vis_train.values, y_vis_train.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 100),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 10))
plt.contourf(X1, X2, classifier4.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Training set)')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_vis_test.values, y_vis_test.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 100),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 10))
plt.contourf(X1, X2, classifier4.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Test set)')
plt.xlabel('ApplicantIncome')
plt.ylabel('LoanAmount')
plt.legend()
plt.show()

#Bonus: CAP Curve (Cumulative Accuracy Profile)
def CAP(classifier, model_name):
    from sklearn.metrics import auc
    
    # Get predicted probabilities for the positive class (1)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    
    # Combine true values and probabilities into a dataframe
    df_cap = pd.DataFrame({'true': y_test, 'prob': y_prob})
    df_cap = df_cap.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    # Total number of samples and positive cases
    total = len(df_cap)
    total_positives = df_cap['true'].sum()
    
    # X and Y coordinates
    x_vals = np.arange(1, total + 1) / total
    y_vals = np.cumsum(df_cap['true']) / total_positives
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='Model', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Model', color='gray')
    plt.plot([0, total_positives/total, 1], [0, 1, 1], label='Perfect Model', color='green')
    
    # Area Under CAP
    cap_auc = auc(x_vals, y_vals)
    print(f"{model_name} CAP AUC:", cap_auc)
    
    plt.title(f'CAP Curve - {model_name}')
    plt.xlabel('% of Applicants (sorted by predicted risk)')
    plt.ylabel('% of Positive Credit Histories Captured')
    plt.legend()
    plt.grid(True)
    plt.show()

CAP(classifier, "Decision Tree")
CAP(classifier2, "Random Forest")

print(y.value_counts(normalize=True))
