<!--
<h2 align="center">
  Group Project - Kaggle Challenge 
</h2>
<h3 align="center">
7PAM2015-0105--2023-Team1
</h3>
-->
<!-- Intro  -->
<h1 align="center">
        <samp>&gt; Group Project - Kaggle Challenge
          <br>
                <b><a target="_blank" href="https://github.com/7PAM2015-0105-2023-Team1">
7PAM2015-0105--2023-Team1</a></b>
        </samp>
</h1>

<p align="center"> 
<p>
<div style="text-align:center;">
    <img src="Kaggle challenge (2).jpg" alt="Spaceship Titanic" style="width:100%;" />
</div>


<div style="text-align:center;">
    <h2 style="font-size: 24px;">
        <samp>
            <a href="https://www.kaggle.com/competitions/spaceship-titanic/">「Spaceship Titanic」</a>
            <br>
            「Presented to: Robert Yates」
        </samp>
    </h2>
</div>



<!-- About Section -->
 # Group Members
 ### Features
- Arsalan Khan
- Muhammad Umair
- Ejaz Malik
- Ehsan Tabassum
- Muhammad Hanzala
- Muhammad Hamza Naseer

<br/>

## Tools Used in the Project

[![Jupyter Notebook Badge](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Python Version](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![Kaggle Profile Badge](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com/your_username)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-green?logo=github)


 ##### Python Libraries
![pandas](https://img.shields.io/badge/pandas-1.3.3-blue?logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.21.2-blue?logo=numpy)
![matplotlib](https://img.shields.io/badge/matplotlib-3.4.3-blue?logo=matplotlib)
![seaborn](https://img.shields.io/badge/seaborn-0.11.2-blue?logo=seaborn)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue?logo=scikit-learn)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-blue?logo=pytorch)

##### Input Data
![CSV file](https://img.shields.io/badge/CSV_File-0170FE?style=for-the-badge&logo=antdesign&logoColor=white)

##### Output Data
[![Predicted Transported Passengers](https://img.shields.io/badge/Predicted%20Transported%20Passengers-1000+-blue)](https://yourlinkhere.com)
[![EDA](https://img.shields.io/badge/EDA-brightgreen)](https://yourlinkhere.com)
[![Score](https://img.shields.io/badge/Score-0.79-red)](https://yourlinkhere.com)


## Overview
- The code is written in a Jupyter Notebook environment, denoted by the .ipynb extension.
- It involves importing various Python libraries such as pandas, numpy, matplotlib, seaborn, scikit-learn, and PyTorch for data manipulation, visualization, and machine learning tasks.
- The objective appears to be tackling a Kaggle challenge related to analyzing data from the Spaceship Titanic incident and predicting whether passengers were transported or not.
- The code includes steps for data preprocessing, exploratory data analysis (EDA), feature engineering, and model building.
- Preprocessing steps involve reading CSV files from URLs, handling missing values, scaling numerical features, and encoding categorical features.
- Exploratory data analysis (EDA) includes visualizations such as histograms, heatmaps, scatter plots, and pair plots to understand the relationships between variables and identify patterns in the data.
- Model building involves defining, training, and evaluating a neural network using PyTorch for binary classification.
- The trained model is evaluated using accuracy score and used to make predictions on test data.
- The predictions are then saved to a CSV file for submission to the Kaggle challenge.


## Working Schema
To describe the model, its successes and failures, and collaborative work on the Kaggle Challenge, we will break down the process into sections as follows:
+ #### Model Description and Architecture:
    + The model used in this project is a neural network architecture implemented using PyTorch.
    + The model architecture consists of multiple layers including input, hidden, and output layers.
    + Specifically, the SpaceshipTitanic class defines the architecture with input features, hidden layers, and output layer.
    + Each layer is connected via linear transformation followed by activation functions like ReLU and Sigmoid.
    + The model is trained using binary cross-entropy loss and optimized using the stochastic gradient descent (SGD) optimizer.
    + The model architecture facilitates learning patterns from the input data to predict the binary outcome, which in this case is whether a passenger was transported.
  
+ #### Training Scheme:
    + The model is trained using a training dataset, which is split into training and testing sets.
    + Training is performed over multiple epochs, where the model is trained to minimize the loss function.
    + The optimizer updates the model parameters based on the gradients of the loss function with respect to the parameters.
    + Training progress is monitored by printing the loss value at regular intervals during training.
    + After training, the model is evaluated using the testing dataset to measure its performance.

+ #### Exploratory Data Analysis (EDA):
    + Exploratory Data Analysis (EDA) is performed on the training dataset to understand the data and identify patterns.
    + EDA involves statistical analysis, visualization, and exploration of relationships between variables.
    + Various plots such as pair plots, histograms, and categorical plots are used to visualize the data distribution and relationships.
    + Correlation matrices are computed to identify relationships between numerical variables.

+ #### Performance Analysis:
    + Model performance is evaluated using accuracy score on the testing dataset.
    + Accuracy score measures the proportion of correctly predicted outcomes.
    + The accuracy score helps assess how well the model generalizes to unseen data.
    + Additionally, predictions are made on the testing dataset and submitted to Kaggle for evaluation.

+ #### Collaborative Work:
    + Collaborative work involves multiple team members contributing to different aspects of the project.
    + Responsibilities may include data preprocessing, feature engineering, model development, and performance evaluation.
    + Collaboration often requires communication, sharing of ideas, and coordination to ensure smooth progress.
    + Tools like Git/GitHub may be used for version control and collaboration management.

By following this process, the team can effectively describe the model, its successes and failures, and collaborative efforts in completing the Kaggle Challenge. Additionally, the presentation of EDA and performance analysis helps provide insights into the data and model performance.


title: Kaggle Challenge Team 1 Sequence Diagram

participant: Kaggle_Challenge_Team_1.ipynb
participant: pandas
participant: numpy
participant: matplotlib
participant: seaborn
participant: sklearn.preprocessing
participant: sklearn.model_selection
participant: tensorflow
participant: torch
participant: shap

Kaggle_Challenge_Team_1.ipynb -> pandas: read_csv('train.csv')
pandas -> Kaggle_Challenge_Team_1.ipynb: train_df

Kaggle_Challenge_Team_1.ipynb -> pandas: read_csv('test.csv')
pandas -> Kaggle_Challenge_Team_1.ipynb: test_df

Kaggle_Challenge_Team_1.ipynb -> pandas: read_csv('sample_submission.csv')
pandas -> Kaggle_Challenge_Team_1.ipynb: sample_submission_df

Kaggle_Challenge_Team_1.ipynb -> seaborn: heatmap(correlation)
seaborn --> Kaggle_Challenge_Team_1.ipynb: Correlation Heatmap

Kaggle_Challenge_Team_1.ipynb -> seaborn: pairplot(dropped_train_df)
seaborn --> Kaggle_Challenge_Team_1.ipynb: Pair Plot

Kaggle_Challenge_Team_1.ipynb -> seaborn: relplot(x='Age', y='Spa', hue='HomePlanet')
seaborn --> Kaggle_Challenge_Team_1.ipynb: Relation Plot

Kaggle_Challenge_Team_1.ipynb -> seaborn: histplot(dropped_train_df['Age'])
seaborn --> Kaggle_Challenge_Team_1.ipynb: Histogram Plot

Kaggle_Challenge_Team_1.ipynb -> seaborn: displot(dropped_train_df['HomePlanet'])
seaborn --> Kaggle_Challenge_Team_1.ipynb: Distribution Plot

Kaggle_Challenge_Team_1.ipynb -> seaborn: catplot(x='Age', kind='box')
seaborn --> Kaggle_Challenge_Team_1.ipynb: Categorical Plot

Kaggle_Challenge_Team_1.ipynb -> torch: SpaceshipTitanic()
torch --> Kaggle_Challenge_Team_1.ipynb: Model Architecture

Kaggle_Challenge_Team_1.ipynb -> torch: nn.BCELoss()
torch --> Kaggle_Challenge_Team_1.ipynb: Loss Function

Kaggle_Challenge_Team_1.ipynb -> torch: optim.SGD()
torch --> Kaggle_Challenge_Team_1.ipynb: Optimizer

Kaggle_Challenge_Team_1.ipynb -> torch: model(X_train)
torch --> Kaggle_Challenge_Team_1.ipynb: Model Prediction

Kaggle_Challenge_Team_1.ipynb -> sklearn.metrics: accuracy_score(y_test, predicted)
sklearn.metrics --> Kaggle_Challenge_Team_1.ipynb: Accuracy Score

Kaggle_Challenge_Team_1.ipynb -> pandas: read_csv('test.csv')
pandas -> Kaggle_Challenge_Team_1.ipynb: test_df

Kaggle_Challenge_Team_1.ipynb -> torch: model(eval_data)
torch --> Kaggle_Challenge_Team_1.ipynb: Prediction on Test Data

Kaggle_Challenge_Team_1.ipynb -> pandas: DataFrame
pandas --> Kaggle_Challenge_Team_1.ipynb: Submission DataFrame

Kaggle_Challenge_Team_1.ipynb -> pandas: to_csv('submission.csv')
pandas --> Kaggle_Challenge_Team_1.ipynb: CSV Submission


