📖 Project Description

The Sonar vs Mine Prediction project is a machine learning application that classifies sonar signals into two categories: Mine (M) or Rock (R).

The dataset contains 208 samples. Each sample has 60 numerical features that represent the strength of sonar signals reflected at different frequencies. The target labels are:

M → Mine (metal object)

R → Rock (cylindrical rock)

This project demonstrates the end-to-end machine learning workflow, including:

Data preprocessing: normalization, label encoding, and splitting into training/testing sets

Model training: Logistic Regression, Random Forest, and Support Vector Machines (SVM)

Evaluation: measuring accuracy and comparing different models

Deployment-ready pipeline: saving and loading models with pickle for real-time predictions

Prediction interface: a simple command-line program to classify new input values

The trained model achieves an accuracy of around 85–90%, depending on the chosen algorithm and train/test split.

This project serves as a beginner-friendly introduction to building and deploying classification models in Python, while also providing a practical use case for sonar-based detection.
