import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

# Set page configuration
st.set_page_config(page_title="Telco Customer Churn Analysis and Prediction")

@st.cache_data
def load_data():
    data = pd.read_csv("telco_customer_churn.csv")
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(inplace=True)

    # Set the Customer ID as index
    data.set_index('customerID', inplace=True)

    # Keep original categorical values for visualizations
    original_data = data.copy()

    # Convert tenure from months to years for visualization
    original_data['TenureGroup'] = pd.cut(original_data['tenure'], 
                                          bins=[0, 12, 24, 36, 48, 60, float('inf')], 
                                          labels=["0-1 Year", "1-2 Years", "2-3 Years", "3-4 Years", "4-5 Years", "5+ Years"])

    # Create bins for Monthly Charges
    original_data['MonthlyChargesRange'] = pd.cut(original_data['MonthlyCharges'], 
                                                  bins=[0, 30, 60, 90, 120, float('inf')],
                                                  labels=["0-30", "30-60", "60-90", "90-120", "120+"])

    # Encode categorical columns for training (but keep original for visualization)
    categorical_cols = ['Contract', 'TechSupport', 'OnlineSecurity', 'InternetService',
                        'StreamingMovies', 'PaperlessBilling', 'MultipleLines', 'SeniorCitizen']

    encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

    # Convert target column to numeric for XGBoost
    data['Churn'] = data['Churn'].replace({"No": 0, "Yes": 1})

    return data, original_data, encoders

data, original_data, encoders = load_data()

# Prepare data for model training
top_features = ["Contract", "TechSupport", "OnlineSecurity", "InternetService",
                "tenure", "MonthlyCharges", "StreamingMovies", "PaperlessBilling",
                "MultipleLines", "SeniorCitizen"]

# Ensure correct data types
data['tenure'] = data['tenure'].astype(int)
data['MonthlyCharges'] = data['MonthlyCharges'].astype(float)
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)

X = data[top_features]
y = data['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model(X_train, y_train):
    param_grid = {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 6, 10]
    }

    model = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
                         param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    model.fit(X_train, y_train)

    return model.best_estimator_

model = train_model(X_train, y_train)

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an analysis", 
    ["Interactive Data Exploration", "Customer Demographics", "Service Usage vs Churn", "Churn Prediction", "Real-Time Churn Prediction"]
)

# 📊 **Interactive Data Exploration**
if option == "Interactive Data Exploration":
    st.title("🔍 Interactive Data Exploration")
    st.markdown("""
    This section allows you to interactively explore the dataset by filtering data based on selected features. You can:
    
    - Select features to filter the data.
    - View the filtered data.
    - Display summary statistics for the filtered data.
    - View the correlation matrix for numerical columns in the filtered data.
    """)

    # Select features for filtering
    selected_features = st.multiselect(
        "Select features to filter data",
        original_data.columns.tolist()
    )

    filtered_data = original_data.copy()

    # Add sliders for numerical fields
    if 'tenure' in selected_features:
        tenure_min, tenure_max = st.slider("Select Tenure Range (Months)", min_value=0, max_value=72, value=(0, 72))
        filtered_data = filtered_data[(filtered_data["tenure"] >= tenure_min) & (filtered_data["tenure"] <= tenure_max)]
        selected_features.remove('tenure')

    if 'MonthlyCharges' in selected_features:
        charges_min, charges_max = st.slider("Select Monthly Charges Range", min_value=0, max_value=150, value=(0, 150))
        filtered_data = filtered_data[(filtered_data["MonthlyCharges"] >= charges_min) & (filtered_data["MonthlyCharges"] <= charges_max)]
        selected_features.remove('MonthlyCharges')

    if 'TotalCharges' in selected_features:
        total_min, total_max = st.slider("Select Total Charges Range", min_value=0, max_value=int(filtered_data['TotalCharges'].max()), value=(0, int(filtered_data['TotalCharges'].max())))
        filtered_data = filtered_data[(filtered_data["TotalCharges"] >= total_min) & (filtered_data["TotalCharges"] <= total_max)]
        selected_features.remove('TotalCharges')

    for feature in selected_features:
        unique_values = filtered_data[feature].unique().tolist()
        filter_values = st.multiselect(f"Filter {feature}", unique_values, default=unique_values)
        filtered_data = filtered_data[filtered_data[feature].isin(filter_values)]

    st.write("Filtered Data", filtered_data)

    # Display summary statistics (excluding categorical columns)
    numeric_filtered_data = filtered_data.select_dtypes(include=[np.number])
    numeric_filtered_data = numeric_filtered_data.drop(columns=['SeniorCitizen'])  # Exclude SeniorCitizen
    st.write("Summary Statistics", numeric_filtered_data.describe())

    # Display correlation matrix
    st.write("Correlation Matrix")
    if numeric_filtered_data.empty:
        st.write("No numeric data to display correlation matrix.")
    else:
        corr_matrix = numeric_filtered_data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# 📊 **Customer Demographics Visualizations**
elif option == "Customer Demographics":
    st.title("📊 Customer Demographics Analysis")
    st.markdown("""
    In this section, you can visualize the distribution of various demographic features of the customers. You can:
    
    - Select a demographic feature such as Gender, Senior Citizen, Partner, Dependents, or Tenure (Years).
    - View a pie chart showing the distribution of the selected demographic feature among customers.
    """)

    # Select feature for visualization
    feature = st.selectbox(
        "Select a demographic feature",
        ["Gender", "Senior Citizen", "Partner", "Dependents", "Tenure (Years)"]
    )

    # Define mapping for column names
    feature_mapping = {
        "Gender": "gender",
        "Senior Citizen": "SeniorCitizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "Tenure (Years)": "TenureGroup"  # Use grouped tenure
    }

    selected_feature = feature_mapping[feature]

    # Fix labels for Senior Citizen
    if selected_feature == "SeniorCitizen":
        labels = ["No", "Yes"]
        sizes = original_data[selected_feature].value_counts().sort_index().values
    else:
        labels = original_data[selected_feature].value_counts().index
        sizes = original_data[selected_feature].value_counts().values

    # Create Pie Chart
    st.subheader(f"📌 {feature} Distribution")
    fig, ax = plt.subplots()
    colors = sns.color_palette("coolwarm", len(labels))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    ax.set_title(f"{feature} Distribution")
    st.pyplot(fig)

# 📊 **Service Usage Trends vs Churn**
elif option == "Service Usage vs Churn":
    st.title("📊 Service Usage Trends vs Churn")
    st.markdown("""
    This section helps you analyze the impact of different service usage features on customer churn. You can:
    
    - Select a service feature such as Contract, TechSupport, OnlineSecurity, InternetService, Tenure (Years), Monthly Charges, StreamingMovies, PaperlessBilling, MultipleLines, or SeniorCitizen.
    - View a count plot showing the churn rate by the selected service feature.
    """)

    # Dropdown for selecting top features
    feature = st.selectbox(
        "Select a service feature to analyze churn impact",
        ["Contract", "TechSupport", "OnlineSecurity", "InternetService", "Tenure (Years)",
         "Monthly Charges", "StreamingMovies", "PaperlessBilling", "MultipleLines", "SeniorCitizen"]
    )

    # Map selected feature to correct column name
    feature_mapping = {
        "Tenure (Years)": "TenureGroup",
        "Monthly Charges": "MonthlyChargesRange"
    }

    selected_feature = feature_mapping.get(feature, feature)  # Use mapped name if available

    st.subheader(f"📌 Churn Rate by {feature}")

    # Ensure selected feature exists before plotting
    if selected_feature not in original_data.columns:
        st.error(f"Error: Column {selected_feature} not found in data.")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=selected_feature, hue="Churn", data=original_data, ax=ax, palette="viridis")
        ax.set_title(f"{feature} Effect on Churn")
        st.pyplot(fig)

# 🤖 **Churn Prediction Model**
elif option == "Churn Prediction":
    st.title("🔮 Real-Time Churn Prediction using XGBoost")
    st.markdown("""
    This section provides insights into the performance of the churn prediction model. You can:
    
    - View the accuracy, F1 score, precision, and recall of the model.
    - View the confusion matrix for the model's predictions.
    - View the ROC curve and AUC score for the model.
    - View the feature importance plot showing the most important features used by the model.
    """)

    # Model Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Model Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.success(f"Model Accuracy: {accuracy:.2%}")
    st.info(f"F1 Score: {f1:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual No Churn", "Actual Churn"], columns=["Predicted No Churn", "Predicted Churn"])
    st.write(cm_df)

    # Plot ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Feature Importance Plot
    st.subheader("Feature Importance (XGBoost)")
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index, palette="viridis")
    ax.set_title("Feature Importance (XGBoost)")
    st.pyplot(fig)

# 🤖 **Real-Time Churn Prediction**
elif option == "Real-Time Churn Prediction":
    st.title("🔮 Predict Churn for a New Customer")
    st.markdown("""
    This section allows you to predict churn for a new customer in real-time. You can:
    
    - Choose to enter customer details manually or enter a Customer ID from the dataset.
    - If entering details manually, provide values for various features such as Contract Type, Tech Support, Online Security, Internet Service, Tenure, Monthly Charges, Streaming Movies, Paperless Billing, Multiple Lines, and Senior Citizen.
    - If entering a Customer ID, the app retrieves the details for that customer from the dataset.
    - View the churn prediction result and churn probability for the new customer.
    """)

    # User Inputs for Model Prediction
    st.sidebar.subheader("Enter Customer Details")

    input_type = st.sidebar.radio("Choose input method", ("Enter Customer ID", "Enter Details Manually"))

    if input_type == "Enter Customer ID":
        customer_id = st.sidebar.text_input("Enter Customer ID")

        if st.sidebar.button("Predict Churn"):
            if customer_id in original_data.index:
                customer_data = original_data.loc[customer_id].to_frame().T
                for col in ["Contract", "TechSupport", "OnlineSecurity", "InternetService", "StreamingMovies", "PaperlessBilling", "MultipleLines"]:
                    customer_data[col] = encoders[col].transform(customer_data[col])
                
                # Ensure correct data types
                customer_data['tenure'] = customer_data['tenure'].astype(int)
                customer_data['MonthlyCharges'] = customer_data['MonthlyCharges'].astype(float)
                customer_data['SeniorCitizen'] = customer_data['SeniorCitizen'].astype(int)

                prediction = model.predict(customer_data[top_features])[0]
                churn_probability = model.predict_proba(customer_data[top_features])[0][1]

                churn_result = "Yes, this customer is likely to churn." if prediction == 1 else "No, this customer is likely to stay."
                st.subheader("🔍 Churn Prediction Result")
                st.success(f"Prediction: {churn_result}")
                st.info(f"Churn Probability: {churn_probability:.2%}")
            else:
                st.error("Customer ID not found.")

    else:
        contract = st.sidebar.selectbox("Contract Type", original_data["Contract"].unique())
        tech_support = st.sidebar.selectbox("Tech Support", original_data["TechSupport"].unique())
        online_security = st.sidebar.selectbox("Online Security", original_data["OnlineSecurity"].unique())
        internet_service = st.sidebar.selectbox("Internet Service", original_data["InternetService"].unique())
        tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, step=1)
        monthly_charges = st.sidebar.slider("Monthly Charges", min_value=0, max_value=150, step=1)
        streaming_movies = st.sidebar.selectbox("Streaming Movies", original_data["StreamingMovies"].unique())
        paperless_billing = st.sidebar.selectbox("Paperless Billing", original_data["PaperlessBilling"].unique())
        multiple_lines = st.sidebar.selectbox("Multiple Lines", original_data["MultipleLines"].unique())
        senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])

        # Convert user inputs into a DataFrame
        user_data = pd.DataFrame({
            "Contract": [contract],
            "TechSupport": [tech_support],
            "OnlineSecurity": [online_security],
            "InternetService": [internet_service],
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "StreamingMovies": [streaming_movies],
            "PaperlessBilling": [paperless_billing],
            "MultipleLines": [multiple_lines],
            "SeniorCitizen": [1 if senior_citizen == "Yes" else 0]  # Convert "Yes"/"No" to 1/0
        })

        # Encode categorical values to match training data
        for col in ["Contract", "TechSupport", "OnlineSecurity", "InternetService", "StreamingMovies", "PaperlessBilling", "MultipleLines"]:
            user_data[col] = encoders[col].transform(user_data[col])

        # Ensure correct data types
        user_data['tenure'] = user_data['tenure'].astype(int)
        user_data['MonthlyCharges'] = user_data['MonthlyCharges'].astype(float)
        user_data['SeniorCitizen'] = user_data['SeniorCitizen'].astype(int)

        # Predict Churn
        if st.sidebar.button("Predict Churn"):
            prediction = model.predict(user_data)[0]
            churn_probability = model.predict_proba(user_data)[0][1]

            # Show Results
            churn_result = "Yes, this customer is likely to churn." if prediction == 1 else "No, this customer is likely to stay."
            st.subheader("🔍 Churn Prediction Result")
            st.success(f"Prediction: {churn_result}")
            st.info(f"Churn Probability: {churn_probability:.2%}")