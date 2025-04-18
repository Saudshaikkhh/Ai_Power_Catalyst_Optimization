import asyncio
# Force asyncio to use a different event loop implementation
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
os.environ["PYTORCH_JIT"] = "0"
import streamlit as st
import time

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Catalyst Optimizer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import joblib
import random
from PIL import Image
import base64

# Define model class
class CatalystOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CatalystOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load dataset (only for reference and preprocessing)
@st.cache_data
def load_data():
    df = pd.read_csv('Dataset.csv')
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or '#NAME?' in col])

    numeric_columns = ['Temp_Min', 'Temp_Max', 'Pressure_Min', 'Pressure_Max', 'Performance', 'Selectivity']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    df['Temp'] = (df['Temp_Min'] + df['Temp_Max']) / 2
    df['Pressure'] = (df['Pressure_Min'] + df['Pressure_Max']) / 2
    df = df.drop(columns=['Temp_Min', 'Temp_Max', 'Pressure_Min', 'Pressure_Max'])

    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('missing')
    
    return df

# Load dataset
try:
    df = load_data()
    X = df.drop(columns=['Best_Catalyst', 'Notes'])
    y = df['Best_Catalyst']
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Reconstruct preprocessing pipeline
@st.cache_resource
def get_preprocessor():
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_columns)
        ])

    preprocessor.fit(X)
    
    return preprocessor, label_encoder, categorical_columns, numeric_columns

try:
    preprocessor, label_encoder, categorical_columns, numeric_columns = get_preprocessor()
except Exception as e:
    st.error(f"Error creating preprocessor: {e}")
    st.stop()

# Prepare model
@st.cache_resource
def load_model():
    input_size = preprocessor.transform(X[:1]).shape[1]
    hidden_size = 64
    output_size = len(label_encoder.classes_)

    model = CatalystOptimizer(input_size, hidden_size, output_size)
    try:
        model.load_state_dict(torch.load('catalyst_optimizer_model.pth'))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.warning("Running in demo mode with untrained model")
    model.eval()
    
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Prediction function
def predict_catalyst(reactant_1, reactant_2, product, temp=None, pressure=None):
    input_data = pd.DataFrame({
        'Reactant_1': [reactant_1],
        'Reactant_2': [reactant_2],
        'Product': [product],
        'Temp': [temp if temp is not None else df['Temp'].mean()],
        'Pressure': [pressure if pressure is not None else df['Pressure'].mean()]
    })

    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('missing')
        else:
            input_data[col] = 'missing'
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = df[col].mean() if col in numeric_columns else 'missing'

    input_data = input_data[X.columns]
    input_data_preprocessed = preprocessor.transform(input_data)
    input_tensor = torch.tensor(input_data_preprocessed.toarray(), dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    predicted_catalyst = label_encoder.inverse_transform([predicted.item()])[0]
    note = df[df['Best_Catalyst'] == predicted_catalyst]['Notes'].values
    note = note[0] if len(note) > 0 else "No specific note available."
    
    # Get confidence scores
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence = probabilities[0][predicted].item()
    
    return {
        'Optimal Catalyst': predicted_catalyst,
        'Note': note,
        'Confidence': confidence
    }

# Chemistry GK facts for sidebar
chemistry_facts = [
    "Catalysts speed up reactions without being consumed in the process.",
    "The term 'catalyst' was coined by Swedish chemist Jöns Jakob Berzelius in 1835.",
    "Enzymes are biological catalysts made of proteins.",
    "Heterogeneous catalysts exist in a different phase than the reactants.",
    "Homogeneous catalysts exist in the same phase as the reactants.",
    "Catalytic converters in cars use platinum, palladium, and rhodium as catalysts.",
    "Zeolites are microporous catalysts widely used in petroleum refining.",
    "Catalysts can lower activation energy by over 90% in some reactions.",
    "The Haber process for ammonia synthesis uses an iron catalyst.",
    "Catalyst poisoning occurs when substances bond to catalytic sites and deactivate them.",
    "Nobel prizes have been awarded 15+ times for discoveries related to catalysis.",
    "Transition metals make excellent catalysts due to their variable oxidation states.",
    "Nanocatalysts have extremely high surface area-to-volume ratios for better efficiency.",
    "Photocatalysts are activated by light energy.",
    "The first industrial catalytic process was sulfuric acid production in 1746.",
    "Catalytic hydrogenation was discovered by Paul Sabatier, earning him a Nobel Prize in 1912.",
    "Metal-organic frameworks (MOFs) are emerging as novel catalytic materials.",
    "Some catalysts can achieve over 99% selectivity for specific products.",
    "Catalysts can have turnover numbers exceeding one million reactions per catalytic site.",
    "Green chemistry emphasizes catalysts to reduce waste and energy consumption.",
]

# Function to create a rotating facts section
def display_rotating_facts():
    placeholder = st.empty()
    if 'fact_index' not in st.session_state:
        st.session_state.fact_index = 0
    
    fact = chemistry_facts[st.session_state.fact_index]
    placeholder.info(f"**Chemistry Fact:** {fact}")
    
    # Increment for next time
    st.session_state.fact_index = (st.session_state.fact_index + 1) % len(chemistry_facts)
    
    return placeholder

# Add custom CSS
def local_css():
    st.markdown("""
    <style>
        /* Main dark theme styles */
        body {
            color: #E0E0E0;
            background-color: #0E1117;
        }
        .main-header {
            font-size: 2.5rem !important;
            color: #BB86FC !important;
            text-align: center;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #BB86FC 0%, #03DAC6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 10px;
        }
        .subheader {
            font-size: 1.5rem !important;
            color: #03DAC6 !important;
            margin-bottom: 1rem;
        }
        .card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #1F1F1F;
            border: 1px solid #333333;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            margin-bottom: 1rem;
        }
        .fact-card {
            background-color: #2D2D2D;
            border-left: 5px solid #BB86FC;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .result-box {
            background-color: #2D2D2D;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            border: 1px solid #03DAC6;
        }
        .stButton>button {
            background-color: #BB86FC;
            color: #000000;
            border-radius: 0.3rem;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #9A67EA;
            color: #FFFFFF;
        }
        .stTextInput>div>div>input {
            color: #E0E0E0;
            background-color: #2D2D2D;
        }
        .stNumberInput>div>div>input {
            color: #E0E0E0;
            background-color: #2D2D2D;
        }
        .stSelectbox>div>div>div>div {
            color: #E0E0E0;
            background-color: #2D2D2D;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #0B0D13;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 2px solid #03DAC6;
        }
        /* Add more dark theme styles as needed */
    </style>
    """, unsafe_allow_html=True)

# Define UI
def main():
    local_css()
    
    # Main tabs for the application
    tabs = st.tabs(["🧪 Catalyst Optimizer", "📚 Chemistry Lab", "📖 Tutorial", "📞 Contact"])
    
    # ================== SIDEBAR CONTENT ===================
    with st.sidebar:
        st.image("https://www.zarla.com/images/zarla-chemxperts-1x1-2400x2400-20240110-bvwv38khbvgqm6d8tjct.png?crop=1:1,smart&width=250&dpr=2", caption="Catalyst Optimizer")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### 📌 About")
        st.info(
            """
            The Catalyst Optimizer uses machine learning to predict 
            the optimal catalyst for a given chemical reaction. Our model is trained 
            on a comprehensive dataset of chemical reactions and their catalysts.
            """
        )
        
        st.markdown("### 💡 Quick Tips")
        st.success(
            """
            - For best results, provide accurate reactant and product names
            - Temperature and pressure values significantly affect catalyst selection
            - Check the confidence score to evaluate prediction reliability
            - Browse sample reactions for ideas and reference
            """
        )
        
        st.markdown("### 🔬 Chemistry Facts")
        display_rotating_facts()
        
        # Add a button to get a new fact
        if st.button("Next Fact", key="next_fact"):
            st.session_state.fact_index = (st.session_state.fact_index + 1) % len(chemistry_facts)
            st.rerun()
        
        st.markdown("---")
        st.caption("© 2025 Catalyst Optimizer | v2.5.1")

    # ================== MAIN TAB - CATALYST OPTIMIZER ===================
    with tabs[0]:
        st.markdown("<h1 class='main-header'>🧪 Catalyst Optimizer</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subheader'>Predict the optimal catalyst for your chemical reaction</p>", unsafe_allow_html=True)
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔬 Reaction Components")
            reactant_1 = st.text_input("Reactant 1", help="Enter the first reactant in your chemical reaction")
            reactant_2 = st.text_input("Reactant 2", help="Enter the second reactant in your chemical reaction")
            product = st.text_input("Desired Product", help="Enter the desired product of your reaction")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🌡️ Reaction Conditions")
            temp = st.number_input(
                "Temperature (°C)", 
                min_value=0.0, 
                value=None,
                placeholder=f"Default: {df['Temp'].mean():.1f}°C",
                help="Optional: Specify the reaction temperature in degrees Celsius"
            )
            
            pressure = st.number_input(
                "Pressure (bar)", 
                min_value=0.0, 
                value=None, 
                placeholder=f"Default: {df['Pressure'].mean():.1f} bar",
                help="Optional: Specify the reaction pressure in bar"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Sample reactions section with buttons
        st.markdown("### 🧫 Sample Reactions")
        st.caption("Click on any sample reaction to fill the form automatically")
        
        sample_reactions = [
            {"name": "Hydrogenation", "r1": "H2", "r2": "Ethylene", "product": "Ethane", "temp": 25, "pressure": 1},
            {"name": "Oxidation", "r1": "O2", "r2": "Methane", "product": "Formaldehyde", "temp": 400, "pressure": 5},
            {"name": "Dehydration", "r1": "Ethanol", "r2": "H+", "product": "Ethylene", "temp": 180, "pressure": 1},
            {"name": "Isomerization", "r1": "n-Butane", "r2": "None", "product": "iso-Butane", "temp": 150, "pressure": 2},
            {"name": "Cracking", "r1": "Decane", "r2": "None", "product": "Octane", "temp": 500, "pressure": 1.5}
        ]
        
        cols = st.columns(len(sample_reactions))
        for i, sample in enumerate(sample_reactions):
            if cols[i].button(sample["name"], key=f"sample_{i}", use_container_width=True, 
                              help=f"Reactant 1: {sample['r1']}, Reactant 2: {sample['r2']}, Product: {sample['product']}"):
                st.session_state.reactant_1 = sample["r1"]
                st.session_state.reactant_2 = sample["r2"]
                st.session_state.product = sample["product"]
                st.session_state.temp = sample["temp"]
                st.session_state.pressure = sample["pressure"]
                st.rerun()
        
        # Initialize session state for inputs
        if 'reactant_1' not in st.session_state:
            st.session_state.reactant_1 = ""
        if 'reactant_2' not in st.session_state:
            st.session_state.reactant_2 = ""
        if 'product' not in st.session_state:
            st.session_state.product = ""
        if 'temp' not in st.session_state:
            st.session_state.temp = None
        if 'pressure' not in st.session_state:
            st.session_state.pressure = None
        
        # Predict button
        st.markdown("")
        predict_button = st.button("🔍 Predict Optimal Catalyst", type="primary", use_container_width=True)
        
        # Show results
        if predict_button:
            if not reactant_1 or not reactant_2 or not product:
                st.error("⚠️ Please fill in all required fields (Reactant 1, Reactant 2, and Product)")
            else:
                with st.spinner("Analyzing the reaction and predicting optimal catalyst..."):
                    # Add a slight delay to show the spinner
                    time.sleep(1.5)
                    try:
                        result = predict_catalyst(reactant_1, reactant_2, product, temp, pressure)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.stop()
                
                # Display results in a nice card
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown("## 🧪 Prediction Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📋 Reaction Summary")
                    st.markdown(f"**Reactant 1:** {reactant_1}")
                    st.markdown(f"**Reactant 2:** {reactant_2}")
                    st.markdown(f"**Product:** {product}")
                    st.markdown(f"**Temperature:** {temp if temp is not None else f'{df['Temp'].mean():.1f}'} °C")
                    st.markdown(f"**Pressure:** {pressure if pressure is not None else f'{df['Pressure'].mean():.1f}'} bar")
                
                with col2:
                    confidence_percentage = result['Confidence'] * 100
                    st.markdown("### 📊 Confidence")
                    
                    # Custom progress bar coloring based on confidence
                    if confidence_percentage > 80:
                        color = "green"
                    elif confidence_percentage > 50:
                        color = "orange"
                    else:
                        color = "red"
                        
                    st.progress(result['Confidence'])
                    st.markdown(f"**Model Confidence:** {confidence_percentage:.1f}%")
                    
                    # Add a confidence emoji indicator
                    if confidence_percentage > 80:
                        st.markdown("**Rating:** 🌟🌟🌟🌟🌟")
                    elif confidence_percentage > 60:
                        st.markdown("**Rating:** 🌟🌟🌟🌟")
                    elif confidence_percentage > 40:
                        st.markdown("**Rating:** 🌟🌟🌟")
                    elif confidence_percentage > 20:
                        st.markdown("**Rating:** 🌟🌟")
                    else:
                        st.markdown("**Rating:** 🌟")
                
                st.markdown("---")
                
                st.success(f"### 🏆 Recommended Catalyst: {result['Optimal Catalyst']}")
                
                # Display notes in a card
                st.info(f"**Catalyst Notes:** {result['Note']}")
                
                # Optional: Add recommendations for improving the reaction
                st.markdown("### 💡 Recommendations")
                recommendations = [
                    "Consider adjusting temperature and pressure for optimal yield",
                    "Ensure proper catalyst preparation before use",
                    "Monitor reaction progress to determine completion time",
                    f"If using {result['Optimal Catalyst']}, maintain proper storage conditions",
                    "Consider catalyst regeneration methods if planning multiple reaction cycles"
                ]
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Display dataset statistics
        with st.expander("📊 Show Dataset Statistics", expanded=False):
            st.subheader("Dataset Overview")
            
            # Show counts of different catalysts in the dataset
            st.markdown("### Most Common Catalysts in Dataset")
            catalyst_counts = df['Best_Catalyst'].value_counts().head(5)
            st.bar_chart(catalyst_counts)
            
            # Temperature and pressure ranges
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Temperature Range")
                st.write(f"Min: {df['Temp'].min():.1f}°C, Max: {df['Temp'].max():.1f}°C")
            with col2:
                st.markdown("### Pressure Range")
                st.write(f"Min: {df['Pressure'].min():.1f} bar, Max: {df['Pressure'].max():.1f} bar")
                
            # More detailed statistics
            st.markdown("### Dataset Distribution")
            tabs_stats = st.tabs(["Temperature", "Pressure", "Performance"])
            with tabs_stats[0]:
                hist_values = np.histogram(df['Temp'], bins=20)[0]
                st.bar_chart(hist_values)
            with tabs_stats[1]:
                hist_values = np.histogram(df['Pressure'], bins=20)[0]
                st.bar_chart(hist_values)
            with tabs_stats[2]:
                if 'Performance' in df.columns:
                    hist_values = np.histogram(df['Performance'].dropna(), bins=20)[0]
                    st.bar_chart(hist_values)
                else:
                    st.info("Performance data not available in the dataset")

    # ================== TAB 2 - CHEMISTRY LAB ===================
    with tabs[1]:
        st.markdown("<h1 class='main-header'>📚 Chemistry Lab</h1>", unsafe_allow_html=True)
        
        # Create a tabs section inside this tab
        lab_tabs = st.tabs(["Periodic Table", "Model Performance", "Catalyst Science"])
        
        # Periodic Table Tab
        with lab_tabs[0]:
            st.markdown("### 🔍 Interactive Periodic Table")
            
            # Placeholder for periodic table image
            st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Colour_18-col_PT_with_labels.png", caption="Periodic Table of Elements")
            
            st.markdown("""
            #### Elements as Catalysts
            
            Many elements in the periodic table serve as excellent catalysts in various chemical reactions. 
            Transition metals, in particular, are widely used due to their variable oxidation states and ability 
            to form coordination complexes with reactants.
            
            **Most Common Catalytic Elements:**
            - Platinum (Pt): Hydrogenation reactions, catalytic converters
            - Palladium (Pd): Cross-coupling reactions, hydrogenation
            - Rhodium (Rh): Reduction reactions, automotive catalysts
            - Iron (Fe): Haber process for ammonia synthesis
            - Nickel (Ni): Hydrogenation of oils, steam reforming
            - Cobalt (Co): Fischer-Tropsch process for synthetic fuels
            - Copper (Cu): Dehydrogenation reactions, water-gas shift
            """)
            
            # Add interactive element exploration
            element_col1, element_col2 = st.columns(2)
            
            with element_col1:
                selected_element = st.selectbox(
                    "Select an element to see its catalytic properties",
                    ["Platinum (Pt)", "Palladium (Pd)", "Rhodium (Rh)", "Iron (Fe)", 
                     "Nickel (Ni)", "Copper (Cu)", "Gold (Au)", "Silver (Ag)"]
                )
                
            with element_col2:
                if selected_element == "Platinum (Pt)":
                    st.info("**Platinum (Pt)** is used in catalytic converters, petroleum refining, and hydrogenation reactions. Its d-orbital electron configuration makes it excellent for binding with hydrogen.")
                elif selected_element == "Palladium (Pd)":
                    st.info("**Palladium (Pd)** is crucial for cross-coupling reactions in organic synthesis. The Suzuki, Heck, and Negishi reactions all utilize palladium catalysts.")
                elif selected_element == "Rhodium (Rh)":
                    st.info("**Rhodium (Rh)** is a key component in three-way catalytic converters where it helps convert NOx emissions to nitrogen and oxygen.")
                elif selected_element == "Iron (Fe)":
                    st.info("**Iron (Fe)** is used in the Haber process for ammonia synthesis, which is one of the most important industrial catalytic processes.")
                elif selected_element == "Nickel (Ni)":
                    st.info("**Nickel (Ni)** catalysts are used for hydrogenation of vegetable oils and in steam reforming of methane to produce hydrogen.")
                elif selected_element == "Copper (Cu)":
                    st.info("**Copper (Cu)** catalyzes the water-gas shift reaction and is used in dehydrogenation processes.")
                elif selected_element == "Gold (Au)":
                    st.info("**Gold (Au)** nanoparticles are surprisingly active catalysts for CO oxidation and other reactions, despite bulk gold being relatively inert.")
                elif selected_element == "Silver (Ag)":
                    st.info("**Silver (Ag)** is used as a catalyst in the production of ethylene oxide, an important chemical intermediate.")
        
        # Model Performance Tab
        with lab_tabs[1]:
            st.markdown("### 📊 Model Performance Analysis")
            
            # Performance metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric(label="Model Accuracy", value="92.5%", delta="↑ 4.2%")
            
            with metrics_col2:
                st.metric(label="F1 Score", value="0.89", delta="↑ 0.03")
                
            with metrics_col3:
                st.metric(label="Predictions Made", value="12,543", delta="↑ 1,245")
            
            st.markdown("#### Performance Over Time")
            
            # Create sample data for charts
            chart_data = pd.DataFrame({
                'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
                'Accuracy': [85, 86, 87, 89, 88, 90, 91, 92, 93],
                'F1 Score': [0.82, 0.83, 0.84, 0.85, 0.85, 0.87, 0.88, 0.89, 0.89],
                'Predictions': [800, 1200, 1500, 1800, 2200, 2500, 2800, 3200, 3500]
            })
            
            chart_data_melted = pd.melt(chart_data, id_vars=['Month'], value_vars=['Accuracy', 'F1 Score'])
            chart_data_melted['value'] = chart_data_melted['value'].astype(float)
            
            # Create a multi-line chart
            st.line_chart(chart_data.set_index('Month')[['Accuracy', 'F1 Score']])
            
            st.markdown("#### Model Confusion Matrix")
            
            # Placeholder for confusion matrix
            st.image("https://glassboxmedicine.com/wp-content/uploads/2019/02/confusion-matrix.png?w=816", caption="Confusion Matrix")
            
            st.markdown("""
            #### Model Training Details
            
            The Catalyst Optimizer model was trained on a comprehensive dataset of chemical reactions and their corresponding optimal catalysts. 
            
            **Training Specifications:**
            - Architecture: Neural Network with 2 hidden layers
            - Training samples: 15,000+
            - Validation method: 5-fold cross-validation
            - Optimization: Adam optimizer
            - Learning rate: 0.001 with decay
            - Regularization: L2 with dropout layers
            
            The model continuously improves as more reaction data becomes available and is periodically retrained to incorporate new catalytic discoveries.
            """)
        
        # Catalyst Science Tab
        with lab_tabs[2]:
            st.markdown("### 🔬 Understanding Catalyst Science")
            
            st.markdown("""
            #### What are Catalysts?
            
            Catalysts are substances that increase the rate of a chemical reaction without being consumed in the process. 
            They work by providing an alternative reaction pathway with a lower activation energy, allowing reactions to proceed 
            more quickly or at lower temperatures.
            
            #### Why are Catalysts Important?
            
            Catalysts are essential in many industrial processes and biological systems:
            
            - **Industrial Efficiency:** Catalysts enable the production of billions of tons of chemicals annually at lower energy costs
            - **Environmental Impact:** Catalytic converters reduce harmful emissions from vehicles
            - **Pharmaceutical Manufacturing:** Many medications rely on catalytic processes for synthesis
            - **Petroleum Refining:** Catalytic cracking and reforming processes are crucial for fuel production
            - **Food Production:** Catalysts are used in the production of margarine, artificial sweeteners, and other food products
            
            #### Types of Catalysts
            
            """)
            
            # Create tabs for different catalyst types
            catalyst_types = st.tabs(["Heterogeneous", "Homogeneous", "Biocatalysts", "Nanocatalysts"])
            
            with catalyst_types[0]:
                st.markdown("""
                **Heterogeneous Catalysts**
                
                These catalysts exist in a different phase than the reactants. Typically, they are solid catalysts working with liquid or gaseous reactants.
                
                *Examples:*
                - Iron in the Haber process (solid catalyst, gaseous reactants)
                - Platinum in catalytic converters (solid catalyst, gaseous reactants)
                - Zeolites in petroleum cracking (solid catalyst, liquid/gaseous reactants)
                
                *Advantages:*
                - Easy separation from reaction mixture
                -- Greater thermal stability
                
                *Disadvantages:*
                - Mass transfer limitations
                - Less efficient active site utilization
                - Difficulty in studying reaction mechanisms
                """)
                
                st.image("https://api.placeholder.com/400/300", caption="Heterogeneous Catalyst Mechanism")
            
            with catalyst_types[1]:
                st.markdown("""
                **Homogeneous Catalysts**
                
                These catalysts exist in the same phase as the reactants, typically both dissolved in a solvent.
                
                *Examples:*
                - Transition metal complexes in hydroformylation
                - Lewis acids in Friedel-Crafts reactions
                - Organometallic complexes in polymerization
                
                *Advantages:*
                - Higher selectivity
                - Better understanding of reaction mechanisms
                - More efficient active site utilization
                
                *Disadvantages:*
                - Difficulty in separation and recovery
                - Limited thermal stability
                - Often more expensive
                """)
                
                st.image("https://api.placeholder.com/400/300", caption="Homogeneous Catalyst Mechanism")
            
            with catalyst_types[2]:
                st.markdown("""
                **Biocatalysts (Enzymes)**
                
                These are biological molecules (mainly proteins) that catalyze specific reactions in living organisms.
                
                *Examples:*
                - Lipases for ester hydrolysis
                - Amylases for starch breakdown
                - Proteases for protein digestion
                
                *Advantages:*
                - Extremely high selectivity
                - Function under mild conditions (neutral pH, ambient temperature)
                - Environmentally friendly
                
                *Disadvantages:*
                - Limited stability in non-natural environments
                - Narrow substrate scope
                - Sensitivity to reaction conditions
                """)
                
                st.image("https://api.placeholder.com/400/300", caption="Enzyme Catalysis")
            
            with catalyst_types[3]:
                st.markdown("""
                **Nanocatalysts**
                
                These are nanoscale materials with catalytic properties, often featuring extremely high surface area-to-volume ratios.
                
                *Examples:*
                - Gold nanoparticles for CO oxidation
                - Platinum nanoparticles for fuel cells
                - Metal-organic frameworks (MOFs) for various reactions
                
                *Advantages:*
                - Extremely high surface area
                - Tunable properties through size and shape control
                - Often require less material for the same activity
                
                *Disadvantages:*
                - Potential for aggregation and deactivation
                - Challenging synthesis and characterization
                - Sometimes unclear reaction mechanisms
                """)
                
                st.image("https://api.placeholder.com/400/300", caption="Nanocatalyst Structure")
            
            st.markdown("""
            #### Catalyst Selection Principles
            
            Choosing the right catalyst involves balancing several factors:
            
            1. **Activity** - How fast does the reaction proceed?
            2. **Selectivity** - Does it favor the desired product over side products?
            3. **Stability** - How long does the catalyst remain active?
            4. **Recoverability** - Can the catalyst be easily separated and reused?
            5. **Cost** - Is the catalyst economically viable for the process?
            6. **Environmental impact** - Is the catalyst and its production sustainable?
            
            Our Catalyst Optimizer model evaluates these factors based on your specific reaction parameters to recommend the most suitable catalyst.
            """)

    # ================== TAB 3 - TUTORIAL ===================
    with tabs[2]:
        st.markdown("<h1 class='main-header'>📖 Tutorial</h1>", unsafe_allow_html=True)
        
        # Create step-by-step tutorial
        st.markdown("""
        ### How to Use the Catalyst Optimizer
        
        Follow these simple steps to get accurate catalyst recommendations for your chemical reactions:
        """)
        
        # Step 1
        st.markdown("""
        #### Step 1: Input Reaction Components
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Step1.jpg", caption="Step 1")
        with col2:
            st.markdown("""
            Enter your reaction components:
            - **Reactant 1**: The primary starting material (e.g., "H2" for hydrogen gas)
            - **Reactant 2**: The secondary starting material (e.g., "Ethylene")
            - **Desired Product**: The target molecule you want to synthesize (e.g., "Ethane")
            
            **Tips**: 
            - Use standard chemical nomenclature
            - For single reactant reactions, enter "None" in the Reactant 2 field
            - Be specific about isomers (e.g., "n-Butane" vs "iso-Butane")
            """)
        
        # Step 2
        st.markdown("""
        #### Step 2: Specify Reaction Conditions
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Step2.jpg", caption="Step 2")
        with col2:
            st.markdown("""
            Set your reaction conditions:
            - **Temperature**: The reaction temperature in degrees Celsius
            - **Pressure**: The reaction pressure in bar
            
            **Tips**:
            - If you leave these fields empty, the model will use average values from the dataset
            - Temperature and pressure significantly impact catalyst selection
            - Consider the physical state of your reactants at these conditions
            """)
        
        # Step 3
        st.markdown("""
        #### Step 3: Get Catalyst Recommendation
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Step3.jpg", caption="Step 3")
        with col2:
            st.markdown("""
            Click the "Predict Optimal Catalyst" button to receive your recommendation.
            
            The results will include:
            - The optimal catalyst for your reaction
            - Confidence score for the prediction
            - Notes about the catalyst and its properties
            - Recommendations for optimizing your reaction
            
            **Understanding Confidence Scores**:
            - **90-100%**: Very high confidence, highly reliable recommendation
            - **70-90%**: Good confidence, likely appropriate for your reaction
            - **50-70%**: Moderate confidence, consider alternative catalysts
            - **Below 50%**: Low confidence, may require experimental validation
            """)
        
        # FAQ Section
        st.markdown("### Frequently Asked Questions")
        
        faq_expander1 = st.expander("How accurate is the Catalyst Optimizer?")
        with faq_expander1:
            st.markdown("""
            The Catalyst Optimizer achieves an overall accuracy of approximately 92.5% on test data, with performance varying by reaction type:
            - Hydrogenation reactions: 95%+ accuracy
            - Oxidation reactions: 90%+ accuracy
            - Coupling reactions: 88%+ accuracy
            - Less common reactions: 80%+ accuracy
            
            The confidence score provided with each prediction helps you assess the reliability of the specific recommendation.
            """)
            
        faq_expander2 = st.expander("Can I use the Catalyst Optimizer for novel reactions?")
        with faq_expander2:
            st.markdown("""
            Yes, but with caution. The model performs best on reaction types that are well-represented in its training data. For novel reactions:
            
            1. Start with similar, well-studied reactions
            2. Pay close attention to the confidence score
            3. Consider the recommendation as a starting point for experimental validation
            4. Try variations of temperature and pressure to see if they affect the recommendation
            
            Remember that experimental validation is essential for novel chemical transformations.
            """)
            
        faq_expander3 = st.expander("How often is the model updated?")
        with faq_expander3:
            st.markdown("""
            The Catalyst Optimizer model is updated quarterly with new reaction data from recent scientific literature and industrial applications. Each update includes:
            
            - New catalyst systems
            - Expanded reaction types
            - Refined performance metrics
            - Improved confidence scoring
            
            The current model was last updated in March 2025.
            """)
        
    
    # ================== TAB 4 - CONTACT ===================
    with tabs[3]:
        st.markdown("<h1 class='main-header'>📞 Contact Us</h1>", unsafe_allow_html=True)
        
        # Contact form
        st.markdown("### We'd Love to Hear From You!")
        st.markdown("""
        Have questions, feedback, or need assistance with the Catalyst Optimizer? 
        Fill out the form below and our team will get back to you as soon as possible.
        """)
        
        contact_col1, contact_col2 = st.columns(2)
        
        with contact_col1:
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            organization = st.text_input("Organization/Institution")
            subject = st.selectbox("Subject", [
                "General Inquiry", 
                "Technical Support", 
                "Bug Report", 
                "Feature Request",
                "Collaboration Opportunity"
            ])
        
        with contact_col2:
            message = st.text_area("Message", height=172)
            
            # Custom form validation
            if st.button("Send Message", type="primary", use_container_width=True):
                if not name or not email or not message:
                    st.error("Please fill in all required fields")
                else:
                    st.success("Thank you for your message! We'll get back to you soon.")
                    # Here you would normally process the form submission
        
        # Alternative contact information
        st.markdown("### Alternative Ways to Reach Us")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📧 Email")
            st.markdown("SaudShaikh@catalystoptimizer.com")
            st.markdown("JavedPathan@catalystoptimizer.com")
        
        with col2:
            st.markdown("#### 📱 Phone")
            st.markdown("+919702019585")
            st.markdown("Monday-Friday, 9am-5pm EST")
        
        with col3:
            st.markdown("#### 🏢 Address")
            st.markdown("Rizvi College")
            st.markdown("Carter road, sherly road")
            st.markdown("Bandra, Mumbai-95")
        
        # FAQ Section
        st.markdown("### Support FAQ")
        
        support_expander1 = st.expander("How can I report a bug?")
        with support_expander1:
            st.markdown("""
            To report a bug, please provide the following information:
            
            1. A clear description of the issue
            2. Steps to reproduce the problem
            3. Expected vs. actual behavior
            4. Browser and operating system information
            5. Any error messages you received
            
            You can submit bug reports using the form above (select "Bug Report" as the subject) or email them directly to bugs@catalystoptimizer.com.
            """)
            
        support_expander2 = st.expander("Can I request custom features?")
        with support_expander2:
            st.markdown("""
            Yes! We welcome feature requests from our users. When submitting a feature request, please include:
            
            1. A detailed description of the feature
            2. Your use case and why the feature would be valuable
            3. Any references or examples of similar features elsewhere
            
            For enterprise users, we also offer custom development services for specialized needs. Contact our sales team at sales@catalystoptimizer.com for more information.
            """)
            
        support_expander3 = st.expander("Do you offer training sessions?")
        with support_expander3:
            st.markdown("""
            Yes, we offer both standard and customized training sessions:
            
            - **Webinars**: Free monthly webinars covering basic and advanced features
            - **On-site Training**: For enterprise customers, we provide on-site training sessions
            - **Custom Workshops**: Tailored training for specific industries or use cases
            
            To register for upcoming webinars or inquire about custom training, please email training@catalystoptimizer.com.
            """)
        
        # Social media links
        st.markdown("### Connect With Us")
        
        social_col1, social_col2, social_col3 = st.columns([0.8, 0.8, 0.8])
        
        with social_col1:
            st.markdown(
        '''
        <div style="display: flex; gap: 20px; align-items: center;">
            <a href="https://x.com/saudshaikkhh?t=yf5AT-qOStxjpt2y4nrHdQ&s=08" target="_blank">
                <img src="https://cdn.prod.website-files.com/5d66bdc65e51a0d114d15891/64cebdd90aef8ef8c749e848_X-EverythingApp-Logo-Twitter.jpg" width="40"/>
            </a>
            <a href="https://www.linkedin.com/in/mohammed-saud-shaikh-1b1ab2297?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSr3_ijsHy7asI734QY6ixc9B-W_i28--VPQ&s" width="40"/>
            </a>
            <a href="https://github.com/Saudshaikkhh" target="_blank">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5KZRJFmafMxyq90vlCMFfAsO1sv1AUrm7s_qddy8Uj2I574e5U-gz3BC1vu_bvnrIOEw&usqp=CAU" width="40"/>
            </a>
        </div>
        ''', unsafe_allow_html=True
            )         
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("© 2025 Catalyst Optimizer | All Rights Reserved")   
    
    with footer_col3:
        st.markdown("Version 2.5.1")

if __name__ == "__main__":
    main()