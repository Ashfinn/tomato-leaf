# Tomato-Net: Comprehensive Benchmark of Deep Learning Models for Tomato Disease Classification 🍅

![tomato-net](https://img.shields.io/badge/Benchmark-Vision%20Models-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Dataset](https://img.shields.io/badge/Images-16,012-orange)
![Models](https://img.shields.io/badge/Models-23-tomato)

## 📖 Overview

Tomato-Net is the most comprehensive benchmarking suite for tomato leaf disease classification, evaluating 23 classical machine learning and modern deep learning architectures. This research provides an extensive analysis of the performance-efficiency trade-off across diverse model families, offering actionable insights for real-world agricultural AI deployment on edge devices.

## 🏆 Key Findings

**MobileViTV2-050** emerges as the optimal model, achieving **99.22% accuracy** with only **1.1M parameters** and **0.15 GFLOPs**, making it perfectly suited for edge deployment. Among classical approaches, **KNN** achieved the highest accuracy at **86.17%**.

## 📊 Complete Performance Summary

### Modern Deep Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | Params (M) | FLOPs (G) | Size (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MobileViTV2-050** | **99.22%** | **99.24%** | **99.22%** | **99.19%** | **1.12** | **0.15** | **4.29** |
| **ConvNeXt-Atto** | 99.38% | 99.37% | 99.38% | 99.37% | 3.38 | 0.44 | 12.89 |
| **MaxViT-Nano** | 99.06% | 99.06% | 99.06% | 99.06% | 14.94 | 1.96 | 57.52 |
| **Xception** | 97.28% | - | - | - | 22.9 | - | - |
| **CoAT-Lite-Tiny** | 98.60% | 98.63% | 98.60% | 98.60% | 5.40 | 0.54 | 20.62 |
| **FastViT-T8** | 97.69% | 97.73% | 97.69% | 97.68% | 3.26 | 0.43 | 12.56 |
| **EfficientNet-B1** | 94.26% | - | - | - | 7.8 | - | - |
| **RegNetY-040** | 97.47% | 97.79% | 97.47% | 97.47% | 19.57 | 2.56 | 74.89 |
| **PoolFormer-S12** | 97.88% | 97.97% | 97.88% | 97.90% | 11.41 | 1.50 | 43.52 |
| **MobileNetV2** | 92.79% | - | - | - | 3.5 | - | - |
| **LeViT-128S** | 97.85% | 97.97% | 97.85% | 97.85% | 7.01 | 0.70 | 27.64 |
| **CrossViT-Tiny** | 97.10% | 97.13% | 97.10% | 97.09% | 6.73 | 0.88 | 25.66 |
| **ShuffleNet** | 90.54% | - | - | - | 1.3 | - | - |
| **EfficientViT-B0** | 95.82% | 95.97% | 95.82% | 95.71% | 2.14 | 0.28 | 8.19 |
| **InceptionV3** | 83.36% | - | - | - | 27.2 | - | - |
| **TNT-Small** | 94.75% | 94.82% | 94.75% | 94.74% | 23.37 | 2.35 | 89.17 |
| **ViT-Small** | 95.10% | 95.36% | 95.10% | 95.11% | 21.67 | 2.17 | 82.66 |
| **ViT-Tiny** | 94.04% | 94.05% | 94.04% | 93.96% | 5.53 | 0.55 | 21.08 |
| **BEiT-Base** | 92.82% | 93.03% | 92.82% | 92.83% | 86.52 | 8.61 | 330.77 |
| **GhostNet** | *Trained* | *Trained* | *Trained* | *Trained* | *Available* | *Available* | *Available* |

### Classical Machine Learning Models

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **K-Nearest Neighbors** | 86.17% | - | - | - |
| **Support Vector Machine** | 75.00% | - | - | - |
| **XGBoost** | 68.00% | - | - | - |
| **Naive Bayes** | *Ensemble* | *Ensemble* | *Ensemble* | *Ensemble* |

*Note: Complete metrics including inference times available in the evaluation_results/ directory*

## 🗂️ Repository Structure

```
tomato-net/
├── models/                          # Pre-trained model weights
│   └── tomato_leaf_ghostnet_best.pth
├── notebooks/                       # Jupyter notebooks for analysis
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   ├── model_training.ipynb        # Model training pipeline
│   ├── metric_evaluation.ipynb     # Performance metrics
│   ├── EfficientB1.ipynb           # EfficientNet experiments
│   ├── inception.ipynb             # InceptionV3 experiments
│   ├── Xception.ipynb              # Xception experiments
│   ├── KNN+NB+XGBoost.ipynb        # Classical ML approaches
│   ├── Mobile_shuffle_efficient.ipynb # MobileNet & ShuffleNet
│   └── bg_removed_inception.ipynb  # Background removal studies
├── scripts/                        # Utility scripts
├── evaluation_results/             # JSON metrics for all models
├── confusion_matrices/             # Visual model performance
├── metrics/                        # Extracted evaluation metrics
├── visualizations/                 # Performance graphs and charts
├── dataset/                        # PlantVillage tomato dataset
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato_Early_blight/
│   ├── Tomato_healthy/
│   ├── Tomato_Late_blight/
│   ├── Tomato_Leaf_Mold/
│   ├── Tomato_Septoria_leaf_spot/
│   ├── Tomato_Spider_mites_Two_spotted_spider_mite/
│   ├── Tomato_Target_Spot/
│   ├── Tomato_Tomato_mosaic_virus/
│   └── Tomato_Tomato.YellowLeaf_Curl_Virus/
├── trained_models/                 # Saved model checkpoints
├── complete_results.csv            # Comprehensive results
├── complete_results.xlsx           # Excel version of results
├── evaluation_report.md            # Detailed analysis report
├── extracted_specific_metrics.txt  # Key metrics extraction
└── requirements.txt                # Python dependencies
```

## 🧠 Models Evaluated

### Deep Learning Models
- **Vision Transformers**: ViT-Tiny, ViT-Small, BEiT, CrossViT, TNT
- **Lightweight CNNs**: MobileViTV2, EfficientViT, ConvNeXt-Atto, FastViT, EfficientNet-B1, MobileNetV2, ShuffleNet, GhostNet
- **Hybrid Architectures**: CoAT-Lite, LeViT, MaxViT, PoolFormer
- **Efficient Networks**: RegNetY, Xception, InceptionV3

### Classical Machine Learning
- K-Nearest Neighbors (KNN): 86.17% accuracy
- Support Vector Machines (SVM): 75.00% accuracy  
- XGBoost: 68.00% accuracy
- Naive Bayes (in ensemble approaches)

## 📈 Dataset Details

- **Total Images**: 16,012 annotated tomato leaf images
- **Classes**: 10 disease categories + healthy leaves
- **Diseases**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Mosaic Virus, Yellow Leaf Curl Virus, Healthy
- **Resolution**: 150×150 pixels (standardized)
- **Train/Test Split**: 80/20 stratified split
- **Augmentation**: Rotation, flipping, color jittering, and scaling

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Installation
```bash
git clone https://github.com/your-username/tomato-net.git
cd tomato-net
pip install -r requirements.txt
```

### Training a Model
```bash
python scripts/train.py --model mobilevitv2_050 --epochs 50 --batch_size 32 --lr 0.001
```

### Evaluating a Model
```bash
python scripts/evaluate.py --model_path trained_models/mobilevitv2_050.pth --data_dir dataset/
```

### Running Inference
```bash
python scripts/predict.py --image_path path/to/leaf_image.jpg --model_name mobilevitv2_050
```

## 🔍 Methodology

### Evaluation Metrics
- **Overall Accuracy**: Percentage of correctly classified images
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)  
- **F1-Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Agreement between predictions and actual labels
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **Efficiency Metrics**: Model size, parameter count, FLOPs, inference time

### Feature Extraction Techniques
Multiple feature extraction techniques were employed across different models:
- Deep Features (VGG16, ResNet50)
- Texture Analysis (GLCM, LBP, Gabor Filters)
- Color Descriptors (RGB, HSV histograms)
- Keypoint Features (SIFT)
- Conformable Polynomials
- Global Average Pooling (ResNet50)
- Hybrid feature ensembles

## 🎯 Results Analysis

Our comprehensive evaluation of 23 models reveals several key insights:

1. **MobileViTV2-050** achieves the best balance of accuracy (99.22%) and efficiency (1.1M parameters)
2. **ConvNeXt-Atto** provides exceptional accuracy (99.38%) with moderate computational requirements
3. **Xception** performs well among traditional architectures (97.28% accuracy)
4. Vision Transformers show strong performance but with higher computational costs
5. **KNN** outperforms other classical ML approaches with 86.17% accuracy
6. Model size doesn't always correlate with accuracy (e.g., MobileViTV2 outperforms larger models)
7. Efficient architectures enable real-time inference on resource-constrained devices

## 🌟 Future Work

- [ ] Quantization benchmarks for further optimization
- [ ] Deployment on edge devices (Raspberry Pi, Jetson Nano)
- [ ] Real-time mobile application development
- [ ] Expansion to additional crop diseases
- [ ] Integration with drone-based monitoring systems
- [ ] Federated learning for privacy-preserving model training
- [ ] Additional studies on background removal techniques
- [ ] Ensemble approaches combining best-performing models

## 🤝 Contributing

We welcome contributions to tomato-net! Please feel free to:
- Submit bug reports and feature requests
- Contribute new model implementations
- Improve documentation and code quality
- Share your deployment experiences and use cases
- Add performance results for new architectures

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use tomato-net in your research, please cite:

```bibtex
@misc{tomato-net2025,
  title={Tomato-Net: Comprehensive Benchmark of Vision Models for Tomato Disease Classification},
  author={Obidur Rahman},
  year={2025},
  url={https://github.com/Ashfinn/tomato-net}
}
```

## 🙏 Acknowledgments

- PlantVillage dataset creators and contributors
- PyTorch and Timm library developers
- Research community for advancing vision architectures
- Contributors to all evaluated model architectures

---

**tomato-net** - Enabling accessible AI for agricultural innovation 🍅🤖

*For complete results, detailed analysis, and reproduction instructions, please explore the notebooks and evaluation_results directory.*