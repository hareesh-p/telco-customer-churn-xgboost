# Telco Customer Churn Analysis and Prediction App

Welcome to the Telco Customer Churn Analysis and Prediction App! This Streamlit application provides various analyses and visualizations to help you understand customer churn and predict it using machine learning models.

## Navigation

The sidebar navigation allows you to choose from the following analyses:

1. **Interactive Data Exploration**
2. **Customer Demographics**
3. **Service Usage vs Churn**
4. **Churn Prediction**
5. **Real-Time Churn Prediction**

### 1. Interactive Data Exploration

This section allows you to interactively explore the dataset by filtering data based on selected features. You can:

- Select features to filter the data.
- View the filtered data.
- Display summary statistics for the filtered data.
- View the correlation matrix for numerical columns in the filtered data.

### 2. Customer Demographics

In this section, you can visualize the distribution of various demographic features of the customers. You can:

- Select a demographic feature such as Gender, Senior Citizen, Partner, Dependents, or Tenure (Years).
- View a pie chart showing the distribution of the selected demographic feature among customers.

### 3. Service Usage vs Churn

This section helps you analyze the impact of different service usage features on customer churn. You can:

- Select a service feature such as Contract, TechSupport, OnlineSecurity, InternetService, Tenure (Years), Monthly Charges, StreamingMovies, PaperlessBilling, MultipleLines, or SeniorCitizen.
- View a count plot showing the churn rate by the selected service feature.

### 4. Churn Prediction

This section provides insights into the performance of the churn prediction model. You can:

- View the accuracy, F1 score, precision, and recall of the model.
- View the confusion matrix for the model's predictions.
- View the ROC curve and AUC score for the model.
- View the feature importance plot showing the most important features used by the model.

### 5. Real-Time Churn Prediction

This section allows you to predict churn for a new customer in real-time. You can:

- Choose to enter customer details manually or enter a Customer ID from the dataset.
- If entering details manually, provide values for various features such as Contract Type, Tech Support, Online Security, Internet Service, Tenure, Monthly Charges, Streaming Movies, Paperless Billing, Multiple Lines, and Senior Citizen.
- If entering a Customer ID, the app retrieves the details for that customer from the dataset.
- View the churn prediction result and churn probability for the new customer.

## How to Use the App

1. **Run the Streamlit App**: Run the app using the Streamlit command:
   ```
   streamlit run app.py
   ```

2. **Navigate Using the Sidebar**: Use the sidebar to navigate between different analyses.

3. **Interact with the App**: Follow the instructions in each section to filter data, visualize distributions, analyze service usage, and predict churn.

## Dataset

The app uses the Telco Customer Churn dataset, which includes various features related to customer demographics, account information, and service usage.

## Model

The churn prediction model is built using the XGBoost classifier, and hyperparameter tuning is performed using GridSearchCV.

## Contact

If you have any questions or feedback, please feel free to contact us.

Happy Analyzing!