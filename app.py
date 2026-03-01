import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("medical_insurance.csv")
# Page Configuration
st.set_page_config(page_title="Medical Insurance Project", page_icon="🏥")


# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🚀 Introduction", "🔍 EDA Visualizations", "🔮 Prediction", "👤 About Me"])


# --- INTRODUCTION PAGE ---
if page == "🚀 Introduction":
    st.title("🏥 Medical Insurance Cost Analysis")
    
    # Adding a relevant image (You can use a URL or a local file path)
    # Using a high-quality placeholder image for medical insurance
    st.image("https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=2011&auto=format&fit=crop", 
             caption="Predicting Healthcare Costs through Data Science", 
             use_column_width=True)

    st.markdown("""
    ### 📖 Overview
    Welcome to the **Medical Insurance Cost Prediction** app. This project aims to analyze 
    individual healthcare data to understand how factors like **age, BMI, smoking habits,** and **region** impact the cost of medical insurance.

    ### 🎯 Project Goals
    * **Exploration:** Identify key drivers of insurance charges.
    * **Visualization:** Use data to tell a visual story.
    * **Prediction:** Use Machine Learning to forecast future costs for new patients.

    ---
    *Use the sidebar to navigate through the data analysis and prediction tools.*
    """)


    # --- EDA Questions ---
elif page == "🔍 EDA Visualizations":
    st.title("📊 Exploratory Data Analysis")
    st.write("Visualizations will appear here...")

    eda_q = st.selectbox("Select a Question to Analyze:", [
        "Q1:-What is the distribution of medical insurance charges?",
        "Q2:-What is the age distribution of the individuals?",
        "Q3:-How many people are smokers vs non-smokers?",
        "Q4:-What is the average BMI in the dataset?",
        "Q5:-Which regions have the most number of policyholders?",
        "Q6:-How do charges vary with age?",
        "Q7:-Is there a difference in average charges between smokers and non-smokers?",
        "Q8:-Does BMI impact insurance charges?",
        "Q9:-Do men or women pay more on average?",
        "Q10:-Is there a correlation between the number of children and the insurance charges?",
        "Q11:-How does smoking status combined with age affect medical charges?",
        "Q12:-What is the impact of gender and region on charges for smokers?",
        "Q13:-How do age, BMI, and smoking status together affect insurance cost?",
        "Q14:-Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers",
        "Q15:-Are there outliers in the charges column? Who are the individuals paying the highest costs?",
        "Q16:-Are there extreme BMI values that could skew predictions?",
        "Q17:-What is the correlation between numeric features like age, BMI, number of children, and charges?",
        "Q18:-Which features have the strongest correlation with the target variable (charges)?"
    ])

# Question 1: Univariate Analysis 
    if eda_q == "Q1:-What is the distribution of medical insurance charges?":
        st.subheader("Distribution of Insurance Charges")
        fig, ax = plt.subplots()
        sns.histplot(df['charges'], kde=True, ax=ax, color='blue')
        st.pyplot(fig)
        st.write("Insight: There may be outliers in the charges column.") 

# Question 2: 
    elif eda_q == "Q2:-What is the age distribution of the individuals?":
        st.subheader("📊 Age Distribution of individuals")
    
        # Plotting the histogram using Plotly for interactivity
        fig = px.histogram(df, x="age", nbins=20, 
                       title="Distribution of Age",
                       labels={'age': 'Age (Years)'},
                       color_discrete_sequence=['#636EFA'],
                       marginal="box") # Adds a box plot on top to show quartiles
    
        st.plotly_chart(fig, use_container_width=True)

        # Statistical Summary Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum Age", int(df['age'].min()))
        col2.metric("Average Age", round(df['age'].mean(), 1))
        col3.metric("Maximum Age", int(df['age'].max()))
        

# Question 3: Bivariate Analysis 
    elif eda_q == "Q3:-How many people are smokers vs non-smokers?":
        st.subheader("Smoker Count")
        fig = px.bar(df['smoker'].value_counts(), title="Smoker Count")
        st.plotly_chart(fig)

# Question 4: 
    elif eda_q == "Q4:-What is the average BMI in the dataset?" :
        st.subheader("The average BMI")
        st.metric("Average BMI", round(df['bmi'].mean(), 2))

# Question 5: 
    elif eda_q == "Q5:-Which regions have the most number of policyholders?":
        st.subheader("Policyholders by Region:")
        fig = px.pie(df, names='region', title="Policyholders by Region")
        st.plotly_chart(fig)

# Question 6: 
    elif eda_q == "How do charges vary with age?":
        st.subheader("📈 Age vs. Medical Insurance Charges")
        fig = px.scatter(df, x="age", y="charges", title="Age vs Charges", trendline="ols")
        st.plotly_chart(fig)
        

# Question 7: Smoker Impact on Charges:
    elif eda_q == "Q7:-Is there a difference in average charges between smokers and non-smokers?":
        st.subheader("🚬 Smoking Status vs. Charges")
        fig = px.box(df, x="smoker", y="charges", color="smoker",
                 title="Comparison of Charges for Smokers vs Non-Smokers",
                 points="all") # Shows individual data points alongside the box
        st.plotly_chart(fig, use_container_width=True)
    
        # Calculating the actual average difference
        avg_diff = df.groupby('smoker')['charges'].mean()
        st.write(f"**Average Cost for Smokers:** ${avg_diff['yes']:,.2f}")
        st.write(f"**Average Cost for Non-Smokers:** ${avg_diff['no']:,.2f}")

# Question 8: BMI Impact on Charges:
    elif eda_q == "Q8:-Does BMI impact insurance charges?":
        st.subheader("⚖️ BMI (Body Mass Index) vs. Charges")
        fig = px.scatter(df, x="bmi", y="charges", title="BMI vs Charges")
        st.plotly_chart(fig)
    
# Question 9:Gender Comparison: 
    elif eda_q == "Q9:-Do men or women pay more on average?":   
        st.subheader("👫 Gender-wise Average Insurance Cost")
        fig = px.bar(df.groupby('sex')['charges'].mean().reset_index(), x='sex', y='charges', title="Avg Charges by Gender")
        st.plotly_chart(fig)

# Question 10: Children vs Charges:
    elif eda_q == "Q10:-Is there a correlation between the number of children and the insurance charges?":
        fig = px.box(df, x="children", y="charges", title="Children vs Charges")
        st.plotly_chart(fig)

# Question 11: Smoking + Age vs Charges:
    elif eda_q == "Q11:-How does smoking status combined with age affect medical charges?":
        st.subheader("👨‍🦳 Smoking Status & Age vs. Charges")
        fig = px.scatter(df, x="age", y="charges", color="smoker", title="Age & Smoking vs Charges")
        st.plotly_chart(fig)

# Question 12: Gender + Region for Smokers:
    elif eda_q == "Q12:-What is the impact of gender and region on charges for smokers?":
        st.subheader("🌍 Regional & Gender Impact (Smokers Only)")
        fig = px.bar(df[df['smoker']=='yes'], x="region", y="charges", color="sex", barmode="group")
        st.plotly_chart(fig)

    # Question 13: Age + BMI + Smoking: 
    elif eda_q == "Q13:-How do age, BMI, and smoking status together affect insurance cost?":
        st.subheader("🧪 Age, BMI, and Smoking Status Interaction")
        fig = px.scatter(df, x="age", y="charges", color="smoker", size="bmi", title="Age, BMI & Smoking")
        st.plotly_chart(fig)

    # Question 14: Obese Smokers (BMI > 30) vs Others:
    elif eda_q == "Q14:-Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers":
        st.subheader("🚨 Comparison: Obese Smokers vs. Healthy Non-Smokers")
        df['group'] = df.apply(lambda x: 'Obese Smoker' if x['bmi']>30 and x['smoker']=='yes' else 'Others', axis=1)
        fig = px.violin(df, x="group", y="charges", color="group", box=True)
        st.plotly_chart(fig)

    # Question 15: Charges Outliers:
    elif eda_q == "Q15:-Are there outliers in the charges column? Who are the individuals paying the highest costs?":
        st.subheader("🕵️ Outlier Detection in Medical Charges")
        fig = px.box(df, y="charges", title="Charges Outliers")
        st.plotly_chart(fig)
        st.write(df.nlargest(5, 'charges')) # Individuals paying highest costs


# Question 16: BMI Outliers:
    elif eda_q == "Q16:-Are there extreme BMI values that could skew predictions?":
        st.subheader("⚖️ BMI Outlier Analysis")
        fig = px.box(df, y="bmi", title="BMI Outliers")
        st.plotly_chart(fig)

# Question 17: Numeric Correlation Matrix:
    elif eda_q ==  "Q17:-What is the correlation between numeric features like age, BMI, number of children, and charges?":
        st.subheader("🔗 Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df[['age', 'bmi', 'children', 'charges']].corr(), annot=True, ax=ax)
        st.pyplot(fig)


# Question 18: Strongest Feature:
    elif eda_q == "Q18:-Which features have the strongest correlation with the target variable (charges)?":
        st.subheader("🎯 Strongest Predictors of Charges")
    
        corr = df[['age', 'bmi', 'children', 'charges']].corr()['charges'].sort_values()
        st.bar_chart(corr)
    


# --- medical insurance model prediction ---
elif page == "🔮 Prediction":
    st.title("🔮🏥 Medical Insurance Cost Prediction")
    st.write("Input your data to get a prediction...")

    # --- 1. Model Load ---
    try:
        model = pickle.load(open('medical_insurance_model.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Error: 'medical_insurance_model.pkl' Not found")
        st.stop()

    # --- 2. User input ---
    Age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ("Female", "Male"))
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ("Yes", "No"))
    region = st.selectbox("Region", ("Southeast", "Southwest", "Northeast", "Northwest"))

    # --- 3. Prediction Button ---
    if st.button("Predict Insurance Cost..."):

        # Mapping (Preprocessing)
        sex_val = 1 if gender == "female" else 0
        smoker_val = 1 if smoker == "yes" else 0
        region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
        region_val = region_map[region]

        # Features array created
        features = np.array([[Age, sex_val, bmi, children, smoker_val, region_val]])
        
        # Make a Prediction
        prediction = model.predict(features)

        # Result display
        # ..2f means two decimal places.
        st.success(f"💰 Estimated Cost: ₹ {prediction[0]:,.2f}")

# ---  About Me Page ---
elif page == "👤 About Me":
    st.title("👨‍💻 About Me:")
    st.markdown("""
    Hi, I'm **Rupali Shrinivas Kendre**! 
    I love turning raw data into interactive stories that anyone can understand.
    
    ---
    ### 📍 Contact & Connect
    * **Location:** Nanded, Maharashtra, India (431605)
    * **Email:** [rupalirakh@gmail.com](mailto:rupalirakh@gmail.com)
    * **LinkedIn:** 
    * **GitHub:** 

    *Always open to collaborating on data-driven projects!*
    """)


