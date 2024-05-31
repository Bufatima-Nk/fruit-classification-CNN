# Fruit Classification using Convolutional Neural Networks

## Project Overview

This project aims to automate the classification of fruits using Convolutional Neural Networks (CNNs) on an augmented dataset. Traditional methods for fruit classification are manual and subjective, making them labor-intensive and inconsistent. By leveraging computer vision and image processing techniques, this project enhances fruit classification accuracy and efficiency.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preparation](#data-preparation)
  - [Data Augmentation](#data-augmentation)
  - [Model Development](#model-development)
  - [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Dataset

The dataset used in this project is the Fruits360 collection, a diverse compilation of fruit images. The dataset is organized into subfolders, each representing a distinct fruit category.

- **Source**: [Kaggle Fruits360 Dataset](https://www.kaggle.com/moltean/fruits)
- **Structure**: Subfolder hierarchy with images labeled by fruit type.

## Methodology

### Data Preparation

1. **Importing Libraries**: Essential libraries such as Keras, NumPy, Pandas, Matplotlib, and Scikit-learn are imported.
2. **Loading Dataset**: Images and labels are loaded from the Fruits360 dataset.
3. **Filtering Dataset**: The dataset is filtered to focus on specific fruit labels of interest.
4. **Data Splitting**: The dataset is split into training and testing sets.

### Data Augmentation

Data augmentation techniques are applied to increase the diversity of the training data:
- **Rotations**
- **Shifts**
- **Shearing**
- **Zooming**
- **Flipping**

### Model Development

A CNN is developed with the following architecture:
- **Convolutional Layers**: Feature extraction
- **Pooling Layers**: Down-sampling
- **Dense Layers**: Classification
- **Activation Functions**: ReLU and Softmax

### Training and Evaluation

The model is trained using the augmented training set and evaluated on a validation set. Key metrics such as accuracy and loss are monitored to assess performance. The final model is tested on unseen data to demonstrate its classification capabilities.

## Results

The model achieved a high accuracy rate of 98% on the augmented dataset. Training and validation metrics were closely aligned, indicating good generalization. A confusion matrix provided detailed insights into the model's performance across different fruit categories.

## Conclusion

The project successfully demonstrated the efficacy of CNNs in automating fruit classification, achieving high accuracy and robustness through data augmentation and model optimization.

## Future Work

Potential future enhancements include:
- **Advanced Augmentation**: Experimenting with more augmentation techniques.
- **Transfer Learning**: Utilizing pre-trained models to improve performance.
- **Dataset Expansion**: Incorporating additional datasets for broader generalization.

## References

1. Kaggle. (n.d.). Fruit 360 Dataset. Retrieved from [Kaggle](https://www.kaggle.com/moltean/fruits)
2. Chollet, F. (2018). *Deep Learning with Python*. Manning Publications.
3. Brownlee, J. (2019). *Image Data Augmentation with Keras*. Retrieved from [Machine Learning Mastery](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)
4. TensorFlow Documentation. (n.d.). Keras ImageDataGenerator Class. Retrieved from [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
5. TensorFlow Documentation. (n.d.). Convolutional Neural Networks (CNN) Overview. Retrieved from [TensorFlow](https://www.tensorflow.org/tutorials/images/cnn)

## Author

[Bufatima Nurmuhammad kyzy]

---

This README provides a comprehensive overview of the fruit classification project, detailing each step from data preparation to model evaluation and outlining future directions for improvement.
