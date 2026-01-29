YOLO11s Optimization for Fingerprint Minutiae Detection

This repository contains the complete experimental notebooks used in the study “YOLO11s Optimization for Minutiae Detection.”
The project investigates YOLO11 model variants (YOLO11n/s/m/l/x) and proposes architectural and training customizations to accurately detect fingerprint minutiae as small objects for forensic and biometric applications.
Fingerprint minutiae (bifurcations, ridge endings, and centers) are extremely small and sensitive to downsampling. This work reformulates minutiae extraction as a one-stage object detection problem using YOLO11 and evaluates how high-resolution detection heads, feature refinement, and training strategies affect localization performance.
________________________________________
📂 Repository Contents
Baseline Experiments
•	Fingerprint_Nano_Yolo11_Final.ipynb – YOLO11n baseline experiment
•	Fingerprint_Small_Yolo11_Final.ipynb – YOLO11s baseline experiment
•	Fingerprint_Medium_Yolo11_Final.ipynb – YOLO11m baseline experiment
•	Fingerprint_Large_Yolo11_Final.ipynb – YOLO11l baseline experiment
•	Fingerprint_Extra_Large_Yolo11_Final.ipynb – YOLO11x baseline experiment
Training and Data Adjustments
•	Fingerprint_Small_Yolo11_Oversampling_+_Hyperparameters.ipynb – class-balanced oversampling and hyperparameter tuning
Architecture Customizations
•	Fingerprint_Small_Yolo11_P2_Detection_Head_+_Widened_P2_Backbone.ipynb – high-resolution detection head (stride 4)
•	Fingerprint_Small_Yolo11_P3_Level_SPPF_(Mid_Range_Context).ipynb – mid-level context pooling (SPPF)
•	Fingerprint_Small_Yolo11_Deeper_Multi_Scale_Head_(C2f×4).ipynb – deeper multi-scale detection head
•	Fingerprint_Small_Yolo11_YOLO11s_Minutiae_V3_(Full_Integration).ipynb – full integrated architecture
________________________________________
🧪 Experimental Pipeline
All notebooks follow a common pipeline:
1.	Dataset Preparation
o	Validation of image–label consistency
o	Standardization of dataset structure for Ultralytics YOLO
o	Splitting into training and validation sets
2.	Baseline Benchmarking
o	Evaluation of YOLO11n, YOLO11s, YOLO11m, YOLO11l, and YOLO11x
o	Selection of YOLO11s as the best speed–accuracy trade-off
3.	Model Customization
o	Addition of a high-resolution detection head (P2, stride 4)
o	Strengthening shallow backbone features for ridge micro-texture
o	Enhanced multi-scale feature fusion
o	Mid-level context pooling to suppress ridge-noise false positives
4.	Training Strategy
o	AdamW optimizer with cosine learning-rate schedule
o	Conservative augmentations preserving ridge geometry
o	Loss reweighting to emphasize localization accuracy
o	Mixed-precision training (AMP)
5.	Evaluation
o	COCO-style metrics: mAP@0.50, mAP@0.50:0.95, precision, recall, F1-score
o	Class-wise performance analysis (bifurcation, ending, center)
o	Qualitative visualization of detected minutiae
________________________________________
📊 Dataset
This project uses the Minutiae Leple Dataset:
https://universe.roboflow.com/khubab-ahmad/minutiae-leple
•	520 fingerprint images
•	7,253 annotated minutiae
•	Classes: bifurcation, ridge ending, center
•	Annotations in YOLO bounding-box format
⚠️ The dataset is not included in this repository due to size limitations and must be downloaded separately.
________________________________________
⚙️ Requirements
•	Python 3.9 or higher
•	Ultralytics YOLO (YOLO11)
•	PyTorch
•	OpenCV
•	NumPy
•	Matplotlib
Typical installation:
pip install ultralytics torch opencv-python numpy matplotlib
________________________________________
🚀 How to Run (Google Colab Recommended)
1.	Open any notebook in Google Colab
2.	Upload the dataset ZIP when prompted
3.	Run cells sequentially:
o	Dataset preparation
o	Training
o	Evaluation
o	Visualization
Each notebook is self-contained and reproduces one experiment from the paper.
________________________________________
📈 Results Summary
Key findings from the experiments:
•	YOLO11s achieved the best baseline performance among YOLO11 variants
•	Adding a high-resolution detection head significantly improved recall for minutiae-scale targets
•	Increasing model depth and complexity did not improve performance and often reduced accuracy
•	Mid-level context pooling improved precision but reduced recall
•	The best trade-off configuration was the P2 detection head with a widened shallow backbone
These results highlight the importance of preserving high-resolution spatial features for reliable minutiae localization.
________________________________________
📌 Notes
•	Datasets and trained weights are intentionally excluded
•	All experiments were conducted on Google Colab with GPU support
•	Results correspond to the reported values in the associated paper
•	Notebook names map directly to the ablation experiments

