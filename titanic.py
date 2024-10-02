
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


titanic_data = pd.read_csv('C:/Users/shant/OneDrive/Desktop/vscode/python/datasets/titanic.csv', encoding='ISO-8859-1')




columns_to_drop = [col for col in titanic_data.columns if 'zero' in col]
titanic_data_cleaned = titanic_data.drop(columns=columns_to_drop)


titanic_data_cleaned = titanic_data_cleaned.rename(columns={
    'Passengerid': 'PassengerID',
    '2urvived': 'Survived',
    'Sex': 'Gender',
    'sibsp': 'Siblings/Spouses',
    'Pclass': 'PassengerClass',
    'Embarked': 'EmbarkedPort'
})


titanic_data_cleaned['Age'].fillna(titanic_data_cleaned['Age'].median(), inplace=True)
titanic_data_cleaned['EmbarkedPort'].fillna(titanic_data_cleaned['EmbarkedPort'].mode()[0], inplace=True)


print("Missing values after cleaning:")
print(titanic_data_cleaned.isnull().sum())


print("First few rows of cleaned data:")
print(titanic_data_cleaned.head())




gender_survival_rate = titanic_data_cleaned.groupby('Gender')['Survived'].mean()
print("Survival rate by gender:")
print(gender_survival_rate)


sns.barplot(x='Gender', y='Survived', data=titanic_data_cleaned)
plt.title('Survival Rate by Gender')
plt.xticks([0, 1], ['Male', 'Female'])
plt.show()


class_survival_rate = titanic_data_cleaned.groupby('PassengerClass')['Survived'].mean()
print("Survival rate by passenger class:")
print(class_survival_rate)


sns.barplot(x='PassengerClass', y='Survived', data=titanic_data_cleaned)
plt.title('Survival Rate by Passenger Class')
plt.show()


print("Age distribution of passengers:")
sns.histplot(titanic_data_cleaned['Age'], kde=False)
plt.title('Age Distribution of Passengers')
plt.show()


print("Fare distribution of passengers:")
sns.histplot(titanic_data_cleaned['Fare'], kde=True)
plt.title('Fare Distribution')
plt.show()
