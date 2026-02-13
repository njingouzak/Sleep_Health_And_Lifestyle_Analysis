# **Sleep Disorder Detection and Lifestyle Analysis through Machine Learning**

---
## **Overview**
This project explores the Sleep Health and Lifestyle Dataset, which contains sleep- and lifestyle-related information for 374 individuals. The dataset provides insights into sleep patterns, daily habits, and the presence of sleep disorders such as insomnia and sleep apnea. The analysis focuses on key lifestyle and health indicators, including physical activity levels, stress, and body mass index (BMI), to understand their relationship with sleep health. By examining these factors together, the project highlights how lifestyle behaviors may influence sleep quality and overall well-being.

A machine learning model is developed to predict the presence of sleep disorders based on a combination of sleep-related and lifestyle variables. This predictive approach supports early identification of individuals at risk and demonstrates the potential of data-driven methods to assist in preventive healthcare and personalized interventions. Unlike many previous studies that rely heavily on subjective, self-reported sleep assessments, this project adopts a multifaceted analytical strategy that integrates multiple variables simultaneously. The results enable the identification of trends, correlations, and key predictors associated with sleep duration, sleep quality, and sleep disorders.

---
## **Key Objectives**
- Explore relationships between lifestyle factors and sleep health
- Identify patterns and correlations affecting sleep duration and quality
- Build and evaluate a machine learning model to predict sleep disorders
- Demonstrate the application of data science techniques in healthcare analytics

---
## **Dataset**

The Sleep Health and Lifestyle Dataset comprises 374 rows and 13 columns, covering a wide range of variables related to sleep and daily habits. It contain the following attributes:  

- `Person ID`: An identifier for each individual.
- `Gender`: The gender of the person (Male/Female).
- `Age`: The age of the person in years.
- `Occupation`: The occupation or profession of the person.
- `Sleep Duration`: The number of hours the person sleeps per day.
- `Quality of Sleep`: A subjective rating of the quality of sleep, ranging from 1 to 10.
- `Physical Activity`: The number of minutes the person engages in physical activity daily.
- `Stress Level`: A subjective rating of the stress level experienced by the person, ranging from 1 to 10.
- `BMI Category`: The BMI category of the person (Normal, Overweight, Obese).
- `Blood Pressure`: The blood pressure measurement of the person, indicated as systolic pressure over diastolic pressure.
- `Heart Rate`: The resting heart rate of the person in beats per minute.
- `Daily Steps`: The number of steps the person takes per day.
- `Sleep Disorder`: The presence or absence of a sleep disorder in the person (None, Insomnia, Sleep Apnea).

---
## **Tools and Technologies**
- Python
- Pandas
- Numpy
- Scikit-Learn
- Jupyter Notebook
- Machine Learning

----
## **Steps Involved**

**1- Data preparation and cleaning:** Identifying and addressing duplicates, missing values and data types  
**2- Exploratory data analysis:** descriptive statistic, visualization and correlation matrix    
**3- Data transforming and preprocessing:** Encoding Categorical Variables and Scaling Features using Min-Max Scaler  
**4- Splitting Dataset into Training and Testing Sets:** The dataset is split into 'x_train' and 'x_test' for features, along with 'y_train' and 'y_test' for the corresponding target variable.  
**5- K-Nearest Neighbors (KNN) Model Training and Evaluation:** The KNN model is trained on the training data (x_train and y_train) and utilized to make predictions (Y_pred) on the testing data (x_test).   Evaluation metrics such as accuracy, precision, recall, and F1-score are computed to assess the model's performance.  
**6- Naive Bayes Classifier Model Training and Evaluation:** The Gaussian Naive Bayes classifier is trained using the training data (x_train and y_train) and subsequently used to predict sleep disorders on the testing data (x_test). Similar to the previous model, evaluation metrics including accuracy, precision, recall, and F1-score are computed to assess the performance.   
**7- Decision Tree Classifier Model Training and Evaluation:** The Decision Tree Classifier model is trained on the training dataset (x_train and y_train) and then used to predict sleep disorders on the testing dataset (x_test). Subsequently, evaluation metrics such as accuracy, precision, recall, and F1-score are computed to gauge the model's performance.  
**8- Gradient Boosting Classifier Model Training and Evaluation:** The Gradient Boosting model is trained using the training dataset (x_train and y_train) and utilized to predict sleep disorders in the testing dataset (x_test). Evaluation metrics, including accuracy, precision, recall, and F1-score, are calculated to assess the model's performance.   
**9- Support Vector Machine (SVM) Classifier Model Training and Evaluation:** The SVM model is trained on the training dataset (x_train and y_train) and used to predict sleep disorders on the testing dataset (x_test). Similar to previous steps, evaluation metrics including accuracy, precision, recall, and F1-score are computed to assess the SVM model's performance in predicting sleep disorders based on lifestyle factors.  
**10- Hyperparameter Tuning with GridSearchCV for SVM Classifier:** the GridSearchCV method is employed to perform hyperparameter tuning for the Support Vector Machine (SVM) Classifier. The hyperparameters 'C' (regularization parameter), 'kernel' (type of kernel), and 'gamma' (kernel coefficient for 'rbf' kernel) are explored using the provided parameter grid. GridSearchCV utilizes 5-fold cross-validation to find the best combination of hyperparameters that maximizes accuracy.  
**11- Prediction using Best Estimator:** Using the best estimator obtained from the GridSearchCV process to predict the sleep disorder category for new data.  

----
## **Key Findings**

Based on the evaluation of five machine learning models for sleep disorder detection, the following observations have been made:

**- Tuned Support Vector Machine (SVM) is the Optimal Model:** The SVM model, after hyperparameter tuning with GridSearchCV, achieved the highest overall performance with an accuracy and F1-score **exceeding 91.6%**. Its ability to model complex, non-linear relationships in the data (using the rbf kernel) makes it particularly well-suited for this classification task.

**- Ensemble Methods are Highly Effective:** The Gradient Boosting model (ensemble method) also demonstrated exceptional performance, reaching **over 90%** in all metrics. This reinforces the idea that combining multiple weak learners can create a powerful and robust predictive model for this kind of health and lifestyle dataset.

----
## **Conclusion**

In summary, the project successfully demonstrates that machine learning can be a powerful tool for predicting sleep disorders based on lifestyle and health metrics. With careful data preparation and model selection, specifically a **tuned Support Vector Machine**, we can achieve high predictive accuracy, paving the way for potential applications in preventive healthcare and personalized wellness.

----

