# Electronic Music Classification Project

![](https://th.bing.com/th/id/R.8838fba0665a1cf1d2c75d467bebf00b?rik=QID81gRD5jd%2bNQ&pid=ImgRaw&r=0)

## Overview

This project focuses on classifying electronic music tracks using machine learning techniques. The data is sourced from [BeatsDataset](https://www.kaggle.com/datasets/caparrini/beatsdataset), and the model is built using popular Python libraries such as `pandas`, `sklearn`, `matplotlib`, and more.

You can find a detailed implementation of this project in my [Kaggle notebook](https://www.kaggle.com/code/ahmadsaadaldeen/electronic-music-classification).

## Project Workflow

1. **Data Preprocessing**: 
   - The dataset is loaded using `pandas` and processed to clean and prepare the features for model training.
   - Categorical data is handled using the `OneHotEncoder` from `sklearn.preprocessing` and combined with numerical data using `ColumnTransformer` from `sklearn.compose`.

2. **Data Splitting**:
   - The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

3. **Feature Scaling**:
   - The features are scaled using the `StandardScaler` from `sklearn.preprocessing` to ensure that all features contribute equally to the model.

4. **Model Training**:
   - The classification model chosen for this task is the `KNeighborsClassifier` from `sklearn.neighbors`.
   - Model hyperparameters are tuned using cross-validation to achieve optimal performance.

5. **Model Evaluation**:
   - The model's performance is evaluated using accuracy metrics and visualized using `matplotlib.pyplot`.

## Key Libraries Used

- **pandas**: For data manipulation and analysis.
- **KNeighborsClassifier**: A simple yet effective machine learning algorithm used for classification tasks.
- **OneHotEncoder**: For encoding categorical features.
- **matplotlib.pyplot**: For plotting and visualizing data and results.
- **train_test_split**: For splitting the dataset into training and testing subsets.
- **ColumnTransformer**: To apply different preprocessing steps to different columns.
- **sklearn.preprocessing**: Provides preprocessing utilities like scaling and encoding.
- **sklearn.compose**: Helps in combining multiple feature transformations into a single pipeline.

## Dataset

The dataset used in this project is the [BeatsDataset](https://www.kaggle.com/datasets/caparrini/beatsdataset), which contains various features describing electronic music tracks.


## References

- [Kaggle Notebook: Electronic Music Classification](https://www.kaggle.com/code/ahmadsaadaldeen/electronic-music-classification)
- [BeatsDataset on Kaggle](https://www.kaggle.com/datasets/caparrini/beatsdataset)


