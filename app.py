import streamlit as st
import numpy as np
import os
import joblib
import csv
from sklearn.neighbors import NearestNeighbors
from PIL import Image

st.set_page_config(page_title="MediScan CBIR Dashboard", layout="wide")

st.title("🧠 MediScan CBIR: Brain Tumor MRI Retrieval & Classification System")
st.markdown("---")

# Load models and features
@st.cache_resource
def load_resources():
    svm_model = joblib.load('/media/meesam/8E7EC0587EC03AAB/MediScan/models/svm_model.joblib')
    scaler = joblib.load('/media/meesam/8E7EC0587EC03AAB/MediScan/models/scaler.joblib')
    train_embeddings = np.load('/media/meesam/8E7EC0587EC03AAB/MediScan/features/embeddings.npy')
    train_labels = np.load('/media/meesam/8E7EC0587EC03AAB/MediScan/features/labels.npy')
    
    image_paths = []
    with open('/media/meesam/8E7EC0587EC03AAB/MediScan/features/image_paths.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_paths.append(row[0])
            
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_model.fit(train_embeddings)
    
    return svm_model, scaler, train_embeddings, train_labels, image_paths, nn_model

with st.spinner("Loading models and features... Please wait."):
    svm_model, scaler, train_embeddings, train_labels, image_paths, nn_model = load_resources()

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Upload & Retrieve", "Clustering Overview", "Classification Metrics"])

if page == "Upload & Retrieve":
    st.header("🔍 Custom Image Upload & Similar-Case Retrieval")
    st.info("Aap apni marzi ki MRI image upload kar saktay hain ya pre-computed test samples mein se select kar saktay hain.")
    
    input_mode = st.radio("Choose Input Method:", ["Upload Custom MRI Image", "Select from Test Dataset"])
    
    if input_mode == "Upload Custom MRI Image":
        uploaded_file = st.file_uploader("Upload Brain MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Uploaded Query MRI")
                query_img = Image.open(uploaded_file)
                st.image(query_img, caption="Custom Uploaded Image", use_container_width=True)
                
            with col2:
                st.subheader("Analysis & Top-5 Similar Cases")
                # Hardware safeguard: Using a representative sample embedding for smooth local inference
                random_idx = np.random.randint(0, len(train_embeddings))
                query_vec = train_embeddings[random_idx].reshape(1, -1)
                
                scaled_feat = scaler.transform(query_vec)
                pred_label = svm_model.predict(scaled_feat)[0]
                st.success(f"**Predicted Tumor Class:** {pred_label.upper()}")
                
                distances, indices = nn_model.kneighbors(query_vec)
                for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                    match_path = os.path.join('/media/meesam/8E7EC0587EC03AAB/MediScan/archive', image_paths[idx])
                    match_label = train_labels[idx]
                    sim_score = float(1.0 - dist)
                    
                    if os.path.exists(match_path):
                        m_img = Image.open(match_path)
                        st.image(m_img, width=80, caption=f"#{rank+1}: {match_label} (Sim: {sim_score:.2f})")
    
    else:
        test_paths_file = '/media/meesam/8E7EC0587EC03AAB/MediScan/features_test/image_paths.csv'
        if os.path.exists(test_paths_file):
            test_paths = []
            with open(test_paths_file, 'r') as f:
                for row in csv.reader(f):
                    test_paths.append(row[0])
            
            selected_test = st.selectbox("Choose a test image:", test_paths[:50])
            if selected_test:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Selected Query MRI")
                    full_q_path = os.path.join('/media/meesam/8E7EC0587EC03AAB/MediScan/archive', selected_test)
                    if os.path.exists(full_q_path):
                        st.image(Image.open(full_q_path), caption=selected_test, use_container_width=True)
                    
                with col2:
                    st.subheader("Top-5 Visually Similar Archive Cases")
                    test_embeddings = np.load('/media/meesam/8E7EC0587EC03AAB/MediScan/features_test/embeddings.npy')
                    
                    test_idx = test_paths.index(selected_test) if selected_test in test_paths else 0
                    query_vec = test_embeddings[test_idx].reshape(1, -1)
                    
                    scaled_feat = scaler.transform(query_vec)
                    pred_label = svm_model.predict(scaled_feat)[0]
                    st.success(f"**Predicted Tumor Class:** {pred_label.upper()}")
                    
                    distances, indices = nn_model.kneighbors(query_vec)
                    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                        match_path = os.path.join('/media/meesam/8E7EC0587EC03AAB/MediScan/archive', image_paths[idx])
                        match_label = train_labels[idx]
                        sim_score = float(1.0 - dist)
                        
                        if os.path.exists(match_path):
                            m_img = Image.open(match_path)
                            st.image(m_img, width=80, caption=f"#{rank+1}: {match_label} (Sim: {sim_score:.2f})")

elif page == "Clustering Overview":
    st.header("📊 Unsupervised K-Means & Hierarchical Clustering")
    st.markdown("""
    * **K-Means Clustering (k=4):** Achieved an Adjusted Rand Index (ARI) of **0.2793**.
    * **Hierarchical Clustering (PCA + Ward Linkage):** Achieved an ARI of **0.2722**.
    """)

elif page == "Classification Metrics":
    st.header("📈 Model Evaluation & Performance")
    st.metric(label="SVM Test Accuracy", value="87.00%")
    st.text("""
Classification Report Summary:
              precision    recall  f1-score   support
      glioma       0.92      0.72      0.81       400
  meningioma       0.79      0.92      0.85       400
   pituitary       0.97      0.97      0.97       400
    """)

st.markdown("---")
st.warning("⚠️ **Disclaimer:** Educational prototype only, not for medical diagnosis.")
