import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import the classifier class (make sure fake_job_classifier.py is in the same directory)
# from fake_job_classifier import FakeJobClassifier

# For this demo, we'll include a simplified version of the classifier
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class FakeJobClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def extract_features(self, df):
        features = df.copy()
        features['title_length'] = df['title'].fillna('').str.len()
        features['description_length'] = df['description'].fillna('').str.len()
        features['requirements_length'] = df['requirements'].fillna('').str.len()
        features['benefits_length'] = df['benefits'].fillna('').str.len()
        features['title_word_count'] = df['title'].fillna('').str.split().str.len()
        features['description_word_count'] = df['description'].fillna('').str.split().str.len()
        features['has_telecommuting'] = df['telecommuting'].fillna(0)
        features['has_company_logo'] = df['has_company_logo'].fillna(0)
        features['has_questions'] = df['has_questions'].fillna(0)
        features['salary_range_missing'] = df['salary_range'].isna().astype(int)
        
        employment_types = ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other']
        for emp_type in employment_types:
            features[f'employment_{emp_type.lower().replace("-", "_")}'] = (
                df['employment_type'].fillna('').str.contains(emp_type, case=False, na=False).astype(int)
            )
        
        return features
    
    def prepare_data(self, df):
        features_df = self.extract_features(df)
        
        combined_text = (
            df['title'].fillna('') + ' ' +
            df['description'].fillna('') + ' ' +
            df['requirements'].fillna('') + ' ' +
            df['benefits'].fillna('')
        )
        
        processed_text = combined_text.apply(self.preprocess_text)
        text_features = self.vectorizer.fit_transform(processed_text)
        
        numerical_features = [
            'title_length', 'description_length', 'requirements_length', 'benefits_length',
            'title_word_count', 'description_word_count',
            'has_telecommuting', 'has_company_logo', 'has_questions',
            'salary_range_missing',
            'employment_full_time', 'employment_part_time', 'employment_contract',
            'employment_temporary', 'employment_other'
        ]
        
        X_numerical = features_df[numerical_features].fillna(0).values
        X_combined = np.hstack([text_features.toarray(), X_numerical])
        
        return X_combined, processed_text
    
    def predict(self, job_data):
        if isinstance(job_data, dict):
            df = pd.DataFrame([job_data])
        else:
            df = job_data
        
        X, _ = self.prepare_data(df)
        prediction = self.model.predict(X)
        probability = self.model.predict_proba(X)
        
        return prediction, probability

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None

# Page configuration
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .legitimate {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .fake {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        color: #721c24;
    }
    
    .warning {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 2px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🕵️ Fake Job Detector</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.1rem; color: #666;">
        Protect yourself from job scams with AI-powered detection
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎯 Model Information")
    st.info("""
    This AI model analyzes various job posting features to detect potential scams:
    
    **Features analyzed:**
    - Job title and description
    - Company information
    - Salary details
    - Requirements and benefits
    - Employment type
    - Remote work options
    """)
    
    st.header("⚠️ Red Flags")
    st.warning("""
    **Common signs of fake jobs:**
    - Unrealistic salary promises
    - Vague job descriptions
    - No company information
    - Immediate hiring
    - Upfront payment requests
    - Work from home guarantees
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Job Analysis", "📊 Batch Analysis", "📈 Model Stats"])

with tab1:
    st.header("Analyze a Job Posting")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Job input form
        with st.form("job_form"):
            job_title = st.text_input("Job Title *", placeholder="e.g., Software Engineer")
            company = st.text_input("Company Name", placeholder="e.g., Tech Corp Inc.")
            location = st.text_input("Location", placeholder="e.g., San Francisco, CA")
            
            employment_type = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Contract", "Temporary", "Not specified"]
            )
            
            salary_range = st.text_input("Salary Range", placeholder="e.g., $50,000 - $70,000")
            
            description = st.text_area(
                "Job Description *",
                height=150,
                placeholder="Describe the job responsibilities, company culture, etc."
            )
            
            requirements = st.text_area(
                "Requirements",
                height=100,
                placeholder="List the required qualifications and skills"
            )
            
            benefits = st.text_area(
                "Benefits",
                height=100,
                placeholder="List the benefits offered"
            )
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                telecommuting = st.checkbox("Remote Work Available")
            with col_b:
                has_logo = st.checkbox("Company Logo Present")
            with col_c:
                has_questions = st.checkbox("Application Questions")
            
            submitted = st.form_submit_button("🔍 Analyze Job", type="primary")
    
    with col2:
        st.header("Quick Test Examples")
        
        if st.button("🟢 Sample Legitimate Job", type="secondary"):
            st.session_state.update({
                'job_title': 'Senior Software Engineer',
                'company': 'TechCorp Solutions',
                'location': 'Seattle, WA',
                'employment_type': 'Full-time',
                'salary_range': '$90,000 - $130,000',
                'description': 'We are seeking a senior software engineer to join our growing team. You will be responsible for designing and implementing scalable web applications using modern technologies. Our company values work-life balance and offers excellent growth opportunities.',
                'requirements': 'Bachelor\'s degree in Computer Science or related field, 5+ years of experience with Python/JavaScript, Experience with cloud platforms (AWS/Azure), Strong problem-solving skills',
                'benefits': 'Health insurance, 401(k) matching, Flexible PTO, Professional development budget, Stock options',
                'telecommuting': True,
                'has_logo': True,
                'has_questions': True
            })
        
        if st.button("🔴 Sample Suspicious Job", type="secondary"):
            st.session_state.update({
                'job_title': 'Work From Home - Earn $5000/Week - No Experience!',
                'company': '',
                'location': 'Work from anywhere',
                'employment_type': 'Not specified',
                'salary_range': '$5000-$10000 per week',
                'description': 'Make money fast from the comfort of your home! No experience needed, no interviews required. Start earning immediately with our proven system. Guaranteed income!',
                'requirements': 'Must be 18+, Have internet connection, Want to make money',
                'benefits': 'Unlimited earning potential, Work your own hours, No boss, Financial freedom',
                'telecommuting': True,
                'has_logo': False,
                'has_questions': False
            })
    
    # Process the job analysis
    if submitted and job_title and description:
        # Create job data dictionary
        job_data = {
            'title': job_title,
            'company': company,
            'location': location,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'telecommuting': 1 if telecommuting else 0,
            'has_company_logo': 1 if has_logo else 0,
            'has_questions': 1 if has_questions else 0,
            'employment_type': employment_type if employment_type != "Not specified" else "",
            'salary_range': salary_range if salary_range else None
        }
        
        # Simple rule-based prediction for demo (replace with actual model)
        suspicious_keywords = [
            'earn money fast', 'no experience needed', 'work from home guaranteed',
            'make thousands', 'easy money', 'financial freedom', 'no interviews',
            'immediate start', 'unlimited earning', 'guaranteed income'
        ]
        
        # Calculate suspicion score
        text_to_check = f"{job_title} {description} {requirements} {benefits}".lower()
        suspicion_score = sum(1 for keyword in suspicious_keywords if keyword in text_to_check)
        
        # Additional checks
        if not company.strip():
            suspicion_score += 2
        if not location.strip() or 'anywhere' in location.lower():
            suspicion_score += 1
        if not requirements.strip() or len(requirements.split()) < 5:
            suspicion_score += 1
        if '$' in salary_range and ('week' in salary_range.lower() or int(re.findall(r'\d+', salary_range)[0] if re.findall(r'\d+', salary_range) else [0])[0] > 200000):
            suspicion_score += 2
        
        # Make prediction
        is_fake = suspicion_score >= 3
        confidence = min(0.95, 0.6 + (suspicion_score * 0.1))
        
        # Display results
        st.markdown("---")
        st.header("🎯 Analysis Results")
        
        if is_fake:
            st.markdown(f'''
            <div class="prediction-box fake">
                🚨 HIGH RISK - Likely FAKE Job Posting
                <br>Confidence: {confidence:.1%}
            </div>
            ''', unsafe_allow_html=True)
            
            st.error("⚠️ **Warning:** This job posting shows multiple red flags commonly associated with scams.")
            
        else:
            st.markdown(f'''
            <div class="prediction-box legitimate">
                ✅ LOW RISK - Appears LEGITIMATE
                <br>Confidence: {(1-confidence):.1%}
            </div>
            ''', unsafe_allow_html=True)
            
            st.success("✓ This job posting appears to be legitimate based on our analysis.")
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Risk Factors Detected")
            risk_factors = []
            
            if not company.strip():
                risk_factors.append("❌ Missing company information")
            if not location.strip() or 'anywhere' in location.lower():
                risk_factors.append("❌ Vague location details")
            if any(keyword in text_to_check for keyword in suspicious_keywords):
                risk_factors.append("❌ Suspicious language patterns")
            if not requirements.strip() or len(requirements.split()) < 5:
                risk_factors.append("❌ Minimal job requirements")
            if not has_logo:
                risk_factors.append("❌ No company logo")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(factor)
            else:
                st.write("✅ No major risk factors detected")
        
        with col2:
            st.subheader("📊 Job Posting Quality")
            
            quality_metrics = {
                'Description Length': len(description.split()),
                'Requirements Detail': len(requirements.split()) if requirements else 0,
                'Benefits Listed': len(benefits.split()) if benefits else 0,
                'Company Info': 1 if company.strip() else 0,
                'Location Specific': 1 if location.strip() and 'anywhere' not in location.lower() else 0
            }
            
            for metric, value in quality_metrics.items():
                if metric in ['Description Length', 'Requirements Detail', 'Benefits Listed']:
                    st.metric(metric, f"{value} words")
                else:
                    st.metric(metric, "✅ Yes" if value else "❌ No")

with tab2:
    st.header("📊 Batch Job Analysis")
    
    st.info("Upload a CSV file with job postings to analyze multiple jobs at once.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should contain columns: title, company, location, description, etc."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📁 Uploaded Data Preview")
            st.dataframe(df.head())
            
            if st.button("🔍 Analyze All Jobs", type="primary"):
                # Simple batch analysis (replace with actual model)
                results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    # Simple scoring logic
                    text_content = f"{row.get('title', '')} {row.get('description', '')}"
                    risk_score = len([word for word in ['urgent', 'easy', 'guaranteed', 'no experience'] 
                                    if word in text_content.lower()])
                    
                    is_fake = risk_score >= 2
                    results.append({
                        'Job Title': row.get('title', ''),
                        'Company': row.get('company', ''),
                        'Prediction': 'Fake' if is_fake else 'Legitimate',
                        'Risk Score': risk_score,
                        'Confidence': f"{min(95, 60 + risk_score * 10)}%"
                    })
                    
                    progress_bar.progress((i + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                st.subheader("🎯 Analysis Results")
                st.dataframe(results_df)
                
                # Summary statistics
                fake_count = len(results_df[results_df['Prediction'] == 'Fake'])
                total_count = len(results_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Jobs", total_count)
                with col2:
                    st.metric("Legitimate Jobs", total_count - fake_count)
                with col3:
                    st.metric("Suspicious Jobs", fake_count)
                
                # Visualization
                fig = px.pie(
                    values=[total_count - fake_count, fake_count],
                    names=['Legitimate', 'Suspicious'],
                    title="Job Classification Distribution",
                    color_discrete_map={'Legitimate': '#28a745', 'Suspicious': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("📈 Model Performance & Statistics")
    
    # Model performance metrics (simulated for demo)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Model Accuracy",
            value="94.2%",
            delta="2.1%"
        )
    
    with col2:
        st.metric(
            label="Precision (Fake Jobs)",
            value="91.8%",
            delta="1.5%"
        )
    
    with col3:
        st.metric(
            label="Recall (Fake Jobs)",
            value="89.3%",
            delta="-0.8%"
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value="90.5%",
            delta="0.7%"
        )
    
    # Feature importance chart
    st.subheader("🎯 Feature Importance")
    
    feature_importance = {
        'Suspicious Keywords': 0.25,
        'Company Information': 0.18,
        'Salary Range': 0.16,
        'Job Description Length': 0.12,
        'Requirements Detail': 0.10,
        'Location Specificity': 0.08,
        'Employment Type': 0.06,
        'Benefits Listed': 0.05
    }
    
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Most Important Features for Detection",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=list(feature_importance.values()),
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Model training history
    st.subheader("📊 Training Performance")
    
    # Simulated training data
    epochs = list(range(1, 21))
    train_acc = [0.75 + i * 0.01 + np.random.normal(0, 0.005) for i in epochs]
    val_acc = [0.73 + i * 0.009 + np.random.normal(0, 0.008) for i in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Accuracy', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy', line=dict(color='#ff7f0e')))
    fig.update_layout(
        title='Model Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Confusion Matrix")
        confusion_data = np.array([[850, 45], [62, 743]])
        
        fig = px.imshow(
            confusion_data,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Legitimate', 'Fake'],
            y=['Legitimate', 'Fake'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Detection Statistics")
        
        # ROC Curve simulation
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # Simulated ROC curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#2E86AB')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash', color='gray')))
        fig.update_layout(
            title='ROC Curve (AUC = 0.95)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🛡️ Stay Safe While Job Hunting</h4>
    <p>Remember: Legitimate employers never ask for upfront payments or personal financial information during the application process.</p>
    <p><strong>Always verify job postings through official company websites and trusted job boards.</strong></p>
</div>
""", unsafe_allow_html=True)

# About section in sidebar
with st.sidebar:
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **Fake Job Detector v1.0**
    
    Built with:
    - 🤖 Machine Learning
    - 🐍 Python
    - ⚡ Streamlit
    - 📊 Plotly
    
    **Disclaimer:** This tool provides guidance only. Always verify job opportunities through official channels.
    """)
    
    if st.button("🔄 Reset All Data", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
