# driving-style-clustering
Clustering driving behavior using IMU sensor data with machine learning. Includes feature engineering, K-Means, DBSCAN, and interactive visualization via Streamlit to analyze aggressive and normal driving patterns.

## 📌 Overview

This project focuses on analyzing driving behavior using IMU (Inertial Measurement Unit) sensor data and clustering different driving styles using unsupervised machine learning techniques.

The system processes raw accelerometer and gyroscope data, extracts meaningful features, and groups driving patterns into clusters such as **Aggressive** and **Normal driving styles**.

An interactive dashboard is built using Streamlit to visualize and explore the results.

---

## 🎯 Objectives

* Analyze driving behavior using sensor data
* Perform feature engineering on time-series data
* Apply unsupervised clustering algorithms
* Evaluate clustering performance
* Build an interactive visualization dashboard

---

## 🧠 Methodology

### 1. Data Preprocessing

* Raw IMU data (AccX, AccY, AccZ, GyroX, GyroY, GyroZ)
* Sorted using timestamps
* Converted to magnitude features:

  * Acceleration magnitude
  * Gyroscope magnitude

*Dataset has been downloaded from https://www.kaggle.com/datasets/outofskills/driving-behavior?resource=download

---

### 2. Feature Engineering

Data is segmented into fixed-size windows and features are extracted:

* Mean acceleration
* Standard deviation
* Maximum acceleration
* Gyroscope statistics
* Jerk (rate of change of acceleration)
* Energy (driving intensity)
* Harsh braking and sharp turns

---

### 3. Clustering Algorithms

* K-Means (primary model)
* DBSCAN (outlier detection)
* Hierarchical clustering (comparison)

---

### 4. Model Evaluation

* Silhouette Score
* Cluster vs Actual Label comparison
* Feature distribution analysis

---

## 📊 Results

* Successfully separated aggressive and normal driving behaviors
* Achieved meaningful clustering with real-world noisy data
* Identified behavioral patterns using feature statistics

---

## 🌐 Streamlit Dashboard

An interactive dashboard was developed using Streamlit to:

* Visualize clusters (PCA projection)
* Explore cluster distributions
* Analyze driver behavior
* View dataset interactively

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## 📁 Project Structure

```
driving-style-clustering/
│
├── advanced_driving_project.py   # ML pipeline
├── app.py                        # Streamlit dashboard
├── combined_dataset.csv          # Input dataset
├── final_output.csv              # Processed output
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Run ML Pipeline

```
python advanced_driving_project.py
```

### 3. Launch Dashboard

```
python -m streamlit run app.py
```

## Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  

## 📌 Key Insights

* Driving behavior can be effectively clustered using IMU data
* Feature engineering plays a critical role in model performance
* Unsupervised learning can be validated using labeled datasets

---

## 🔮 Future Improvements

* Real-time driving behavior detection
* Deep learning models (LSTM for time-series)
* Deployment on cloud platforms
* Integration with mobile sensor data

---

## 👨‍💻 Author

*Arpita Ramdurg*

---

## ⭐ Acknowledgment

This project demonstrates practical application of machine learning in real-world scenarios such as driver safety, fleet monitoring, and insurance analytics.

---
