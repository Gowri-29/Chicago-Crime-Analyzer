import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header(':blue[_Chicago Crime Analyzer_]')

# Load dataset
df = pd.read_csv('cleaned_crime_data.csv')

# Preprocess the 'Updated On' column
df['Updated On'] = pd.to_datetime(df['Updated On']).apply(lambda x: x.timestamp())

# Define features and target
features = ["Block", "IUCR", "Primary Type", "Description", "Location Description", "District", "Ward", "Community Area", "FBI Code", "Year", "Updated On", "Latitude", "Longitude"]
target = "Arrest"

# Preprocessing steps for the model pipeline
numerical_features = ["IUCR", "District", "Ward", "Community Area", "Year", "Updated On", "Latitude", "Longitude"]
categorical_features = ["Block", "Primary Type", "Description", "Location Description", "FBI Code"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a logistic regression model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
X = df[features]
y = df[target]
model_pipeline.fit(X, y)

# Collect input features from user
input_features = []
for feature in features:
    if feature in categorical_features:
        input_value = st.text_input(f"Enter {feature}:")
    else:
        input_value = st.number_input(f"Enter {feature}:")
    input_features.append(input_value)

# Ensure all inputs are gathered before making a prediction
if st.button("Predict"):
    if all(input_features):
        input_df = pd.DataFrame([input_features], columns=features)
        
        # Handle 'Updated On' feature conversion
        input_df['Updated On'] = pd.to_datetime(input_df['Updated On']).apply(lambda x: x.timestamp())
        
        predicted_score = model_pipeline.predict(input_df)[0]
        st.write("Predicted arrest likelihood:", "Yes" if predicted_score else "No")
    else:
        st.error("Please enter all feature values.")
st.write("""
         Insights and Recommendations:
            Based on the analysis of the crime data, several key insights have been identified:

            Decreasing Trend in Arrests (2005 to 2020):\n

            Insight: There is a noticeable decreasing trend in the number of arrests from 2005 to 2020.
            
            Implications: This trend could indicate several potential factors such as improved policing strategies, changes in crime reporting practices, societal changes, or other external influences like economic conditions.
            
            Recommendations:
            Analyze Further: Conduct a detailed analysis to understand the underlying reasons for the decline. This could involve looking into changes in law enforcement practices, community programs, or socioeconomic factors during this period.
            
            Focus on Sustaining Improvements: If the decrease is due to effective law enforcement strategies, these should be identified and reinforced. Continuous training and support for law enforcement personnel on these effective strategies should be prioritized.
            
            Community Engagement: Increase community engagement and preventative measures to continue this downward trend. Programs that focus on education, job training, and youth engagement could help in maintaining low crime rates.
            \n
            High Number of Arrests Occurred on Streets:

            Insight: The majority of arrests occur on streets.
            
            Implications: This may suggest that street-level crimes are more prevalent or more likely to be reported and lead to arrests.
            
            Recommendations:
            Increase Street Patrols: Implement more street patrols, especially in high-crime areas, to deter criminal activities and improve response times.
            
            Install Surveillance Systems: Enhance surveillance on streets with high crime rates through the use of cameras and other monitoring technologies to deter crime and assist in quicker identification and apprehension of offenders.
            
            Community Policing: Adopt community policing strategies to build trust between law enforcement and the community, encouraging community members to report crimes and collaborate with police.
            \n
            18th Ward has High Count of Arrests:

            Insight: The 18th Ward has the highest count of arrests among all wards.
            
            Implications: This may indicate a concentration of crime in this area, requiring targeted interventions.
            
            Recommendations:
            Targeted Law Enforcement: Focus law enforcement efforts and resources in the 18th Ward. This could include increased patrols, undercover operations, and targeted actions against known crime hotspots.
            
            Community Programs: Implement community-based programs aimed at reducing crime, such as youth engagement initiatives, employment programs, and drug prevention and rehabilitation services.
            
            Infrastructure Improvements: Improve street lighting, community centers, and public spaces to reduce opportunities for crime and enhance community cohesion.
            \n
            High Count of Arrests with FBI Code - 14:

            Insight: The FBI Code 14, which categorizes certain types of crimes, has the highest count of arrests.
            
            Implications: This suggests that crimes categorized under FBI Code 14 are particularly prevalent and may require specialized attention.
            
            Recommendations:
            Specialized Task Forces: Create specialized task forces to focus on the specific crimes categorized under FBI Code 14. This could include dedicated teams with expertise in these crime types.
            
            Public Awareness Campaigns: Launch public awareness campaigns to educate the community about the specific crimes under FBI Code 14 and how to prevent them. Encourage community members to report suspicious activities related to these crimes.
            
            Enhanced Training: Provide specialized training for law enforcement personnel to better understand and address the types of crimes under FBI Code 14. This could include training on investigation techniques, evidence collection, and prosecution processes.""")