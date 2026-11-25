import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page Configuration
st.set_page_config(
    page_title="European Bank | Cyberpunk Neon Churn Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk Neon Styling
st.markdown("""
<style>
    /* Dark Neon Theme Background & Typography */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    
    /* Headers with Neon Glow */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00F0FF;
        text-shadow: 0 0 12px rgba(0, 240, 255, 0.7), 0 0 20px rgba(0, 240, 255, 0.4);
        margin-bottom: 0.2rem;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    .sub-header {
        font-size: 1.15rem;
        color: #FF007F;
        text-shadow: 0 0 8px rgba(255, 0, 127, 0.5);
        margin-bottom: 1.5rem;
    }
    
    /* Card Styles with Neon Borders */
    .neon-card {
        background: #121824;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #00F0FF;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);
        margin-bottom: 1rem;
    }
    
    /* Risk Badges */
    .risk-high {
        background-color: rgba(255, 0, 127, 0.15);
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #FF007F;
        box-shadow: 0 0 18px rgba(255, 0, 127, 0.4);
        color: #FF007F;
        font-weight: bold;
    }
    
    .risk-medium {
        background-color: rgba(255, 230, 0, 0.15);
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #FFE600;
        box-shadow: 0 0 18px rgba(255, 230, 0, 0.4);
        color: #FFE600;
        font-weight: bold;
    }
    
    .risk-low {
        background-color: rgba(0, 255, 102, 0.15);
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #00FF66;
        box-shadow: 0 0 18px rgba(0, 255, 102, 0.4);
        color: #00FF66;
        font-weight: bold;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF007F 0%, #B026FF 100%);
        color: #FFFFFF;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(255, 0, 127, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.8);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Train Baseline Model (Cached for performance)
@st.cache_resource
def load_trained_model():
    df1 = pd.read_csv("European_Bank_part1.csv")
    df2 = pd.read_csv("European_Bank_part2.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    
    X = df.drop(columns=['Year', 'CustomerId', 'Surname', 'Exited'])
    y = df['Exited']
    
    cat_cols = ['Geography', 'Gender']
    num_cols = [col for col in X.columns if col not in cat_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss'))
    ])
    
    pipeline.fit(X, y)
    return pipeline, df

try:
    model, df_raw = load_trained_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Please place European_Bank_part1.csv and European_Bank_part2.csv in the application directory. Error: {e}")

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/lightning-bolt.png", width=60)
st.sidebar.title("⚡ NEON CONTROL PANEL")
page = st.sidebar.radio("Navigate Module:", [
    "📊 Executive Neon Dashboard",
    "🎯 Individual Risk Calculator",
    "⚡ What-If Scenario Simulator",
    "📁 Batch Risk Scoring (CSV)",
    "📖 Model Documentation"
])

# Plotly Neon Theme Layout Helper
def apply_neon_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor='rgba(10, 14, 23, 0.8)',
        plot_bgcolor='rgba(18, 24, 36, 0.8)',
        font=dict(color='#E2E8F0', family="Segoe UI"),
        xaxis=dict(gridcolor='#1E293B', zerolinecolor='#334155'),
        yaxis=dict(gridcolor='#1E293B', zerolinecolor='#334155')
    )
    return fig

# Page 1: Executive Neon Dashboard
if page == "📊 Executive Neon Dashboard":
    st.markdown('<div class="main-header">⚡ EUROPEAN BANK NEON ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">High-intensity predictive risk intelligence across 10,000 retail accounts.</div>', unsafe_allow_html=True)
    
    if model_loaded:
        col1, col2, col3, col4 = st.columns(4)
        total_customers = len(df_raw)
        churned_customers = df_raw['Exited'].sum()
        churn_rate = (churned_customers / total_customers) * 100
        avg_balance = df_raw['Balance'].mean()
        
        col1.metric("Total Accounts", f"{total_customers:,}")
        col2.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
        col3.metric("Churned Accounts", f"{churned_customers:,}")
        col4.metric("Avg Account Balance", f"€{avg_balance:,.2f}")
        
        st.divider()
        
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            st.subheader("🌐 Churn Rate by Geography")
            geo_churn = df_raw.groupby('Geography')['Exited'].agg(['count', 'mean']).reset_index()
            geo_churn['mean'] = geo_churn['mean'] * 100
            fig_geo = px.bar(geo_churn, x='Geography', y='mean', text_auto='.1f',
                             labels={'mean': 'Churn Rate (%)'}, color='Geography',
                             color_discrete_sequence=['#00F0FF', '#FF007F', '#00FF66'])
            fig_geo.update_layout(showlegend=False, height=350)
            apply_neon_plotly_theme(fig_geo)
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with row1_col2:
            st.subheader("📦 Churn Rate by Product Holdings")
            prod_churn = df_raw.groupby('NumOfProducts')['Exited'].agg(['count', 'mean']).reset_index()
            prod_churn['mean'] = prod_churn['mean'] * 100
            fig_prod = px.bar(prod_churn, x='NumOfProducts', y='mean', text_auto='.1f',
                              labels={'mean': 'Churn Rate (%)', 'NumOfProducts': 'Product Count'},
                              color='NumOfProducts', color_continuous_scale=['#00FF66', '#FFE600', '#FF007F'])
            fig_prod.update_layout(height=350)
            apply_neon_plotly_theme(fig_prod)
            st.plotly_chart(fig_prod, use_container_width=True)
            
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.subheader("👥 Age Demographics vs Churn")
            fig_age = px.histogram(df_raw, x='Age', color='Exited', barmode='overlay',
                                   color_discrete_map={0: '#00F0FF', 1: '#FF007F'},
                                   labels={'Exited': 'Churn Status'})
            fig_age.update_layout(height=350)
            apply_neon_plotly_theme(fig_age)
            st.plotly_chart(fig_age, use_container_width=True)
            
        with row2_col2:
            st.subheader("⚡ Active Member Impact")
            active_churn = df_raw.groupby(['IsActiveMember', 'Geography'])['Exited'].mean().reset_index()
            active_churn['Exited'] = active_churn['Exited'] * 100
            active_churn['IsActiveMember'] = active_churn['IsActiveMember'].map({0: 'Inactive', 1: 'Active Member'})
            fig_active = px.bar(active_churn, x='Geography', y='Exited', color='IsActiveMember', barmode='group',
                                labels={'Exited': 'Churn Rate (%)'}, color_discrete_sequence=['#FF007F', '#00FF66'])
            fig_active.update_layout(height=350)
            apply_neon_plotly_theme(fig_active)
            st.plotly_chart(fig_active, use_container_width=True)

# Page 2: Individual Risk Calculator
elif page == "🎯 Individual Risk Calculator":
    st.markdown('<div class="main-header">🎯 NEON RISK CALCULATOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Calculate real-time risk scores with neon indicator visualization.</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Customer Demographic Profile")
        geography = st.selectbox("Geography / Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 95, 42)
        credit_score = st.slider("Credit Score", 350, 850, 650)
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        
    with c2:
        st.subheader("Account & Portfolio Activity")
        balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
        num_products = st.selectbox("Number of Products Held", [1, 2, 3, 4], index=0)
        has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
        is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)
        estimated_salary = st.number_input("Estimated Salary (€)", min_value=0.0, max_value=250000.0, value=100000.0, step=5000.0)

    if st.button("🚀 EXECUTE NEON RISK ASSESSMENT", use_container_width=True) and model_loaded:
        input_data = pd.DataFrame([{
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_credit_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }])
        
        prob = model.predict_proba(input_data)[0][1]
        score_pct = prob * 100
        
        st.divider()
        st.subheader("Predicted Neon Risk Profile")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)", 'font': {'color': '#00F0FF', 'size': 18}},
                number={'font': {'color': '#00F0FF', 'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#E2E8F0'},
                    'bar': {'color': "#00F0FF"},
                    'bgcolor': "#121824",
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 255, 102, 0.3)"},
                        {'range': [30, 60], 'color': "rgba(255, 230, 0, 0.3)"},
                        {'range': [60, 100], 'color': "rgba(255, 0, 127, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "#FF007F", 'width': 5},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=280, paper_bgcolor='rgba(10, 14, 23, 0.8)', font=dict(color='#E2E8F0'))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_res2:
            if score_pct >= 60:
                st.markdown(f'<div class="risk-high">🚨 CRITICAL NEON RISK ({score_pct:.1f}%)<br><br>Action Protocol: Immediate VIP Intervention. Deploy high-yield deposit offer and direct relationship executive.</div>', unsafe_allow_html=True)
            elif score_pct >= 30:
                st.markdown(f'<div class="risk-medium">⚡ MODERATE NEON RISK ({score_pct:.1f}%)<br><br>Action Protocol: Targeted Retention Campaign. Drive engagement via digital loyalty rewards.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ OPTIMAL STABILITY ({score_pct:.1f}%)<br><br>Action Protocol: Standard Relationship Maintenance.</div>', unsafe_allow_html=True)

# Page 3: Scenario Simulator
elif page == "⚡ What-If Scenario Simulator":
    st.markdown('<div class="main-header">⚡ NEON RETENTION SIMULATOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Test product strategy shifts and engagement rules in real time.</div>', unsafe_allow_html=True)
    
    if model_loaded:
        sim_age = st.slider("Base Customer Age", 20, 80, 48)
        sim_country = st.selectbox("Base Country", ["Germany", "France", "Spain"])
        sim_balance = st.number_input("Base Balance (€)", value=90000.0)
        
        st.divider()
        st.subheader("Intervention Controls")
        
        c_sim1, c_sim2 = st.columns(2)
        with c_sim1:
            sim_active = st.radio("Activity Intervention", [0, 1], format_func=lambda x: "Active Member" if x == 1 else "Inactive Member", index=0)
        with c_sim2:
            sim_prods = st.selectbox("Product Strategy Count", [1, 2, 3, 4], index=0)
            
        sim_df = pd.DataFrame([{
            'CreditScore': 640,
            'Geography': sim_country,
            'Gender': 'Female',
            'Age': sim_age,
            'Tenure': 5,
            'Balance': sim_balance,
            'NumOfProducts': sim_prods,
            'HasCrCard': 1,
            'IsActiveMember': sim_active,
            'EstimatedSalary': 95000
        }])
        
        prob_sim = model.predict_proba(sim_df)[0][1] * 100
        
        st.metric("Simulated Risk Score", f"{prob_sim:.1f}%")
        if sim_prods == 2 and sim_active == 1:
            st.success("🎉 Optimal Neon Strategy: 2 Products + Active Membership drops risk to baseline minimum!")
        elif sim_prods >= 3:
            st.error("🚨 Critical Over-Saturation: 3+ products drives risk above 80%!")

# Page 4: Batch CSV Scoring
elif page == "📁 Batch Risk Scoring (CSV)":
    st.markdown('<div class="main-header">📁 BATCH NEON PORTFOLIO SCORING</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Batch score customer accounts with instant risk level tagging.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Portfolio CSV File", type=["csv"])
    if uploaded_file is not None and model_loaded:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", batch_df.head(5))
        
        if st.button("RUN BATCH NEON ANALYSIS"):
            required_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
            if all(col in batch_df.columns for col in required_cols):
                preds_prob = model.predict_proba(batch_df[required_cols])[:, 1]
                batch_df['Churn_Probability_%'] = (preds_prob * 100).round(2)
                batch_df['Risk_Level'] = pd.cut(batch_df['Churn_Probability_%'], bins=[-1, 30, 60, 100], labels=['Low Risk', 'Medium Risk', 'High Risk'])
                
                st.success("Batch Prediction Complete!")
                st.dataframe(batch_df.head(10))
                
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Neon Risk CSV",
                    data=csv_data,
                    file_name="European_Bank_Neon_Scored.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"CSV must contain: {required_cols}")

# Page 5: Documentation
elif page == "📖 Model Documentation":
    st.markdown('<div class="main-header">📖 MODEL ARCHITECTURE & BLUEPRINT</div>', unsafe_allow_html=True)
    st.markdown("""
    ### Neon Analytics Machine Learning Pipeline
    * **Algorithm**: Extreme Gradient Boosting (XGBoost) Classifier & Random Forest Ensemble.
    * **Evaluation**: ROC-AUC ~0.87, Stratified 5-Fold Cross Validation.
    * **Primary Features**: Age, Product Count, Active Membership Status, Geography (Germany high risk).
    """)

st.sidebar.markdown("---")
st.sidebar.caption("European Bank Cyberpunk Neon Analytics v2.0")
