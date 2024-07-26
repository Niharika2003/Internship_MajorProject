# Semiconductor Manufacturing Process Prediction

This project uses various supervised learning models to predict pass/fail outcomes in a semiconductor manufacturing process. The primary goal is to identify the best model based on accuracy and generalization capability.

**_Introduction_**

In semiconductor manufacturing, predicting whether a product will pass or fail quality control tests is crucial for maintaining high yield rates and reducing costs. This project uses sensor data to train supervised learning models to predict the pass/fail outcomes of semiconductor units.

**_Dataset_**

The dataset contains sensor readings from the manufacturing process along with the pass/fail status of each unit. The primary steps in this project include preprocessing the data, performing exploratory data analysis, training and tuning models, and evaluating their performance.

**_Preprocessing_**

Handling Missing Values:

Drop columns with a high percentage of missing values.
Fill remaining missing values with appropriate methods (e.g., mean for numeric columns).

Dropping Irrelevant Columns:

Columns like 'Time' which do not contribute to the prediction task are removed.
python
Copy code
# Preprocessing code snippet
df = df.drop(columns=['Time'])

df = df.fillna(df.mean())

Exploratory Data Analysis (EDA)

Univariate Analysis:

Analyze the distribution of each feature.
Identify potential outliers and patterns.
python

# Univariate analysis code snippet
df['sensor_1'].hist()

Bivariate Analysis:
Analyze relationships between features and the target variable.
Identify correlations and significant predictors.
python

# Bivariate analysis code snippet
pd.plotting.scatter_matrix(df)

Model Training, Testing, and Tuning
Model Selection:

Support Vector Machine (SVM)
Random Forest
Naive Bayes
Training and Hyperparameter Tuning:

Use GridSearchCV for hyperparameter tuning.
Evaluate models using cross-validation.
python

# Model training and tuning code snippet
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC()
params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svm, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)
Model Evaluation:
Compare models based on accuracy, precision, recall, and F1-score.
Select the best model for deployment.
python
Copy code
# Model evaluation code snippet
from sklearn.metrics import classification_report
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
Conclusion and Improvisation
The SVM model was selected as the best model based on its high test accuracy and balanced performance. Potential improvements include feature engineering, handling data imbalance, and exploring ensemble methods.

Installation
Clone the repository and install the necessary dependencies:

bash
Copy code
git clone https://github.com/your-username/semiconductor-prediction.git
cd semiconductor-prediction
pip install -r requirements.txt
Usage
Run the main script to execute the preprocessing, training, and evaluation pipeline:

