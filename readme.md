# Comparative Analysis of Lightweight Vision Models for Tomato Disease Classification: Towards Edge-Deployable Agricultural AI 🍅

This repository provides a comprehensive comparative analysis of various lightweight vision models for the classification of tomato diseases. The primary goal of this research is to identify efficient and accurate models suitable for deployment on edge devices, enabling real-time, in-field disease diagnosis for farmers.

## 📜 Introduction

Tomato cultivation is a significant agricultural practice worldwide, but it is often hampered by various diseases that can lead to substantial yield losses. Early and accurate disease detection is crucial for effective management. This project explores the potential of state-of-the-art lightweight deep learning models to provide a scalable and accessible solution for tomato disease classification, with a special emphasis on their performance on resource-constrained edge devices.

## 📊 Models and Performance

We evaluated a diverse set of lightweight vision models on a custom dataset of tomato leaf images. The performance of each model was assessed based on several key metrics, including accuracy, precision, recall, F1-score, model size, and computational complexity (FLOPs).

| Model Name | Overall Accuracy (%) | Macro-averaged Precision (%) | Macro-averaged Recall (%) | Macro-averaged F1-score (%) | Model Size (M Params) | FLOPs (GFLOPs) | Avg. Inference Time (ms/image) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileViTV2-050** | **99.56** | **99.56** | **99.56** | **99.56** | **1.1** | **0.2** | **91** |
| MaxViT (maxvit\_nano\_rw\_256) | 99.23 | 99.23 | 99.23 | 99.23 | 15.45 | 9.0 | - |
| ConvNeXt-Atto (convnext\_atto) | 99.15 | 99.16 | 99.15 | 99.15 | 3.7 | 1.1 | 85 |
| FastViT (fastvit\_t8) | 98.63 | 98.64 | 98.63 | 98.63 | 4.0 | 1.4 | 90 |
| CoAtNet-Lite (coat\_lite\_tiny) | 98.14 | 98.20 | 98.14 | 98.14 | 5.7 | 3.2 | 93 |
| RegNetY (regnety\_040) | 97.78 | 97.91 | 97.78 | 97.78 | 20.6 | 8.0 | 87 |
| PoolFormer (poolformer\_s12) | 97.07 | 97.17 | 97.07 | 97.09 | 12.0 | 0.86 | 80 |
| CrossViT (crossvit\_tiny\_240) | 95.98 | 96.03 | 95.98 | 95.97 | 8.69 | 2.90 | - |
| EfficientViT (efficientvit\_b0) | 95.85 | 96.02 | 95.85 | 95.73 | 2.14 | 0.1 | 89 |
| TNT (tnt\_s\_patch16\_224) | 95.47 | 95.51 | 95.47 | 95.47 | 23.8 | 10.4 | 60 |
| ViT-Small (vit\_small\_patch16\_224) | 81.29 | 81.3 | 81.29 | 81.29 | 22.1 | 8.6 | 94 |
| LeViT (levit\_128s) | 76.6 | 76.6 | 76.6 | 76.6 | 7.8 | 0.6 | 95 |
| ViT-Tiny (vit\_tiny\_patch16\_224) | 75.81 | 75.81 | 75.81 | 75.81 | 9.7 | 2.2 | 96 |
| BEiT (beit\_base\_patch16\_224) | 85.36 | 85.36 | 85.36 | 85.36 | 81.1 | 25.4 | 97 |
| PVT (pvt\_tiny) | 79.88 | 79.88 | 79.88 | 79.88 | 13.2 | 1.9 | 35 |
| Twins (twins\_pcpvt\_base) | 81.2 | 81.2 | 81.2 | 81.2 | 43.8 | 13.4 | 98 |
| XCiT (xcit\_tiny\_12\_p16\_224) | 82.1 | 82.1 | 82.1 | 82.1 | 6.8 | 2.8 | - |

***Note:** Some inference times are not reported as they are highly dependent on the specific hardware and software environment.*

## 📈 Evaluation Metrics

The models were evaluated using the following standard classification metrics:

  - **Overall Accuracy:** The percentage of correctly classified images out of the total number of images.
    $$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}$$
  - **Macro-averaged Precision:** The average precision for each class, calculated independently and then averaged. It treats all classes equally, regardless of their size.
    $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
  - **Macro-averaged Recall:** The average recall for each class, calculated independently and then averaged. It measures the ability of the model to identify all relevant instances of each class.
    $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
  - **Macro-averaged F1-score:** The harmonic mean of macro-averaged precision and recall, providing a single score that balances both metrics.
    $$\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  - **Model Size (M Params):** The total number of trainable parameters in the model, in millions. A smaller size is generally better for edge deployment.
  - **FLOPs (GFLOPs):** Floating Point Operations per second, in billions. This metric indicates the computational complexity of the model.
  - **Average Inference Time (ms/image):** The average time taken by the model to process a single image on a given hardware platform.

## 🖼️ Dataset

The models were trained and evaluated on the **PlantVillage dataset**, which contains thousands of images of healthy and diseased tomato leaves, categorized into different disease classes. The dataset was preprocessed and augmented to ensure robustness and prevent overfitting.

## 🚀 Getting Started

To replicate the results or use the models for your own research, follow these steps:

### Prerequisites

  - Python 3.8 or higher
  - PyTorch 1.10 or higher
  - torchvision
  - timm (PyTorch Image Models)
  - scikit-learn
  - NumPy
  - Matplotlib

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/tomato-disease-classification.git
    cd tomato-disease-classification
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Training

To train a model, run the `train.py` script with the desired model architecture and hyperparameters:

```bash
python train.py --model_name mobilevitv2_050 --data_dir /path/to/your/dataset --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Evaluation

To evaluate a trained model, use the `evaluate.py` script:

```bash
python evaluate.py --model_path /path/to/your/trained/model.pth --data_dir /path/to/your/test/dataset
```

## 🔬 Results and Discussion

Our analysis reveals that **MobileViTV2-050** stands out as the top-performing model, achieving an impressive **99.56%** overall accuracy while maintaining a very small model size (1.1M parameters) and low computational cost (0.2 GFLOPs). This combination of high accuracy and efficiency makes it an excellent candidate for deployment on edge devices.

Other models like **MaxViT** and **ConvNeXt-Atto** also demonstrated high accuracy but with a larger number of parameters. The results highlight a clear trade-off between model performance and computational requirements, which is a critical consideration for real-world agricultural AI applications.

## 🤝 Contributing

Contributions to this project are welcome\! If you have suggestions for improving the models, adding new architectures, or enhancing the analysis, please feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.