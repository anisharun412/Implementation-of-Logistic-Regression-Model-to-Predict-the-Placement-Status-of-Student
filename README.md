# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Arunsamy D
RegisterNumber: 212224240016 / 24900591
*/
```

```python
import pandas as pd

data = pd.read_csv('Placement_Data.csv')
data1 = data.drop(["sl_no", "salary"], axis=1)


print(data1.isnull().sum())
print()
print("Duplicates:", data1.duplicated().sum())
print()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']:
    data1[col] = le.fit_transform(data1[col])


x = data1.drop("status", axis=1)
y = data1["status"]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)


y_predict = model.predict(x_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:", accuracy_score(y_test, y_predict))
print()

print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print()

print("Classification Report:\n", classification_report(y_test, y_predict))
print()

sample = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
prediction = model.predict(sample)
print("Prediction for sample:", prediction)

```

## Output:

### Null Values:

![image](https://github.com/user-attachments/assets/09c40563-2e65-4b15-b11a-ff4d1266d541)


### Duplicates:

![image](https://github.com/user-attachments/assets/97a523b8-6e88-471f-a258-c691095db7a0)

### Accuracy:

![image](https://github.com/user-attachments/assets/8d3e3bbb-2cc3-40eb-9e97-f705e556b3e5)

### Confusion Matrix:

![image](https://github.com/user-attachments/assets/34cd9369-3086-485c-a6c5-9918f1c1fa18)

### Classification Report:

![image](https://github.com/user-attachments/assets/c9b16cb4-d11b-432a-94c7-96018640dd79)

### Sample Prediction:

![image](https://github.com/user-attachments/assets/483e36c3-137e-4e7a-bd76-e80e2f128f14)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
