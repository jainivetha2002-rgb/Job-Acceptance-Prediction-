⚙️ Project Workflow
1️⃣ Data Cleaning

Standardized column names

Removed duplicate records

Handled missing values using median and mode

Normalized categorical values

2️⃣ Exploratory Data Analysis (EDA)

Placement distribution analysis

Skills match vs placement

Interview scores vs placement

Company tier and experience impact

All plots saved for reuse in reports and dashboards

3️⃣ Feature Engineering

Experience category (fresher / junior / senior)

Academic average & performance band

Interview average & interview level

Improved interpretability and modeling quality

4️⃣ Model Training

Train–test split

Label encoding for categorical variables

Feature scaling using StandardScaler

Random Forest classifier

Performance evaluated using accuracy and classification report

5️⃣ Model Persistence

Saved trained model

Saved encoders and scaler to ensure consistent predictions

6️⃣ Prediction Pipeline

Loads saved artifacts

Safely handles unseen categorical values

Outputs prediction with confidence score

7️⃣ Streamlit KPI Dashboard

Displays recruitment KPIs

Visual insights with charts

Real-time job acceptance prediction

📈 Key KPIs Displayed

Total Candidates

Placement Rate (%)

Not Placed Rate (%)

Average Interview Score

Average Skills Match (%)

High-Risk Candidate Percentage

🖥️ Streamlit Dashboard

The dashboard provides:

Business-level recruitment insights

Interactive visualizations

Live prediction using trained ML model


🛠️ Tech Stack

Python 3.14

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Joblib

Streamlit

📌 Key Learnings

Building a complete ML pipeline

Feature engineering for business interpretability

Model evaluation and persistence

Safe inference handling

Creating KPI dashboards with Streamlit

👤 Author

Jai nivetha R
Machine Learning & Data Science Enthusiast



