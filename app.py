# ================================
# 🚗 FINAL POLISHED STREAMLIT APP
# ================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Driving Style Analysis", layout="wide")

st.title("🚗 Driving Style Clustering Dashboard")
st.markdown("Analyze and visualize driver behavior using ML clustering")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_output.csv")

try:
    df = load_data()
except:
    st.error("❌ Please run advanced_driving_project.py first")
    st.stop()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Controls")

cluster = st.sidebar.selectbox(
    "Select Cluster",
    sorted(df["kmeans_cluster"].unique())
)

# -------------------------------
# Top Metrics
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", len(df))
col2.metric("Clusters", df["kmeans_cluster"].nunique())
col3.metric("Features", df.shape[1] - 2)

# -------------------------------
# Cluster Distribution
# -------------------------------
st.subheader("📊 Cluster Distribution")
st.bar_chart(df["kmeans_cluster"].value_counts())

# -------------------------------
# PCA Visualization
# -------------------------------
st.subheader("📈 PCA Visualization")

features = df.drop(["label", "window_id", "kmeans_cluster",
                    "dbscan_cluster", "hierarchical_cluster"], axis=1)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(features)

fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["kmeans_cluster"])
ax.set_title("PCA Cluster Visualization")
st.pyplot(fig)

# -------------------------------
# Selected Cluster Details
# -------------------------------
st.subheader(f"📌 Cluster {cluster} Details")

filtered = df[df["kmeans_cluster"] == cluster]

st.write("### Summary Statistics")
st.write(filtered.describe())

# -------------------------------
# Label Distribution
# -------------------------------
st.subheader("🧠 Actual Driving Behavior (Labels)")

if "label" in df.columns:
    st.bar_chart(filtered["label"].value_counts())

# -------------------------------
# Raw Data View
# -------------------------------
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Driving Style Clustering Project")