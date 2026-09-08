Lung X-Ray Classification with Deep Neural Networks
Coursework project for Applied AI (COMP534), MSc Data Science & Artificial Intelligence, University of Liverpool.

Overview
A 3-class classification task (Normal / Opacity / Pneumonia) on ~3,500 chest X-ray images. The project compares a small custom CNN against ResNet-18 under three different transfer-learning strategies, to understand how much pretrained ImageNet knowledge transfers to a medical imaging domain.

What's in this repo
code.ipynb — full pipeline: data loading, preprocessing, augmentation, model definitions, training loops, evaluation.
report.pdf — written report covering data management decisions, architecture design, and results.
Note: the dataset itself is not included (coursework-provided, not mine to redistribute). The notebook expects images organised into Normal/, Opacity/, and Pneumonia/ folders.

Approach
Preprocessing: resized to 224x224, ImageNet-statistics normalisation, stratified 5-fold cross-validation.
Augmentation (train only): random horizontal flip, ±5° rotation — chosen for anatomical plausibility, avoiding transforms (e.g. vertical flip, large shear) that would distort real lung anatomy.
Models compared:
Custom CNN (~340K parameters): 4 convolutional blocks with batch norm, max pooling, and a compact classifier head.
ResNet-18: trained from scratch, fully fine-tuned, and as a frozen feature extractor.
Training: Cross-entropy loss, Adam (lr=0.001), 20 epochs per model.
Results
Model	Strategy	Best Val Acc	Params
CustomCNN	Scratch	0.894	~340K
ResNet-18	Scratch	0.892	11.18M
ResNet-18	Fine-Tuning	0.937	11.18M
ResNet-18	Feature Extractor	0.892	1,538
ResNet-18 Fine-Tuning was selected as the best model and evaluated on the held-out test set: 0.932 accuracy, 0.934 macro F1, 0.986 ROC-AUC (macro).

Tech stack
Python, PyTorch, torchvision, NumPy, scikit-learn, Matplotlib, Seaborn.

My contribution
This was a two-person coursework project. Each of us worked independently on our own implementation, then compared results and combined the best-performing pieces into the final report and notebook. My specific work covered:

Data processing (the preprocessing and augmentation pipeline)
Fine-tuning the ResNet-18 models
Optimisation (loss function, optimiser configuration, and hyperparameter choices such as learning rate)
