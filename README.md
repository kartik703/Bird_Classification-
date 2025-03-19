# Fine-Grained Bird Species Classification





A deep learning pipeline for fine-grained visual categorization of 200 bird species using the CUB-200-2011 dataset, achieving state-of-the-art classification accuracy through advanced CNN architectures and transfer learning.

## Introduction 🐦

**Objective**: Accurate classification of 200 bird species from challenging fine-grained visual data.\
**Key Features**:

- Transfer learning with modern CNN architectures
- Comprehensive data augmentation pipeline
- Advanced evaluation metrics (Top-K Accuracy, Confusion Matrix)
- Model interpretability visualization

**Applications**:\
✔ Biodiversity monitoring\
✔ Ecological research\
✔ Wildlife conservation\
✔ Camera trap image analysis

## Dataset 📚

**CUB-200-2011** ([Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)):

- 200 bird species
- 11,788 images
- Annotations:
  - Bounding boxes
  - Part locations
  - Attribute labels
  - Segmentation masks

**Preprocessing**:

- 224×224 center cropping
- ImageNet normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])
- Augmentations:
  - Random horizontal flip (p=0.5)
  - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2)
  - Random rotation (±15°)

## Features ✨

- 🦅 **Advanced Architectures**: ResNet-50/101, EfficientNet-B4, VGG-16
- 🎨 **Data Augmentation**: 15+ transformation techniques
- 📈 **Learning Strategies**:
  - Cosine annealing learning rate
  - Label smoothing regularization
  - Gradient clipping
- 🔍 **Model Interpretation**: Class activation maps (Grad-CAM)
- 📊 **Evaluation Metrics**:
  - Top-1/Top-5 Accuracy
  - Class-wise F1 Scores
  - Confusion Matrix Analysis

## Results 📈

### Performance Comparison

| Model           | Top-1 Acc | Top-5 Acc | Params (M) |
| --------------- | --------- | --------- | ---------- |
| ResNet-50       | 84.2%     | 96.7%     | 25.6       |
| EfficientNet-B4 | 86.9%     | 97.3%     | 19.3       |
| VGG-16          | 78.4%     | 93.1%     | 138.4      |

### Visualization

\
*Confusion matrix for ResNet-50 (diagonal = correct predictions)*

## Future Work 🔮

1. Implement vision transformers (ViT, Swin)
2. Integrate part localization using bounding boxes
3. Develop few-shot learning approach
4. Create web demo with Gradio/FastAPI
5. Quantization for edge deployment

## References 📚

1. Wah et al. (2011) - [CUB-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/papers/CUB_200_2011.pdf)
2. He et al. (2016) - [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
3. Tan & Le (2019) - [EfficientNet](https://arxiv.org/abs/1905.11946)
