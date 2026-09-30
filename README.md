# PPE Compliance Detection with YOLOv8s

A computer vision project that detects four personal protective equipment (PPE) conditions in images and live webcam frames: **Hardhat**, **NO-Hardhat**, **Safety Vest**, and **NO-Safety Vest**. It compares two fine tuning strategies for YOLOv8s, repeats them with YOLO11s, and explores local webcam inference..

## Project overview and problem

Checking PPE use manually is difficult to scale across many workers. This project explores whether an object detector can locate visible hardhats and safety vests, as well as examples labeled as missing that equipment. For each detection, the output is a class name, a bounding box, and a confidence score. This is a learning project and the webcam demo is not a production safety system.

## Dataset and model

- **Source:** [PPE Dataset for YOLOv8 on Kaggle](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8). The original dataset contains 14 classes; this project uses the four PPE classes listed above.
- **Curation:** A 5,000-image subset was selected. After removing 104 exact-image duplicate copies and merging their complementary annotations, the final dataset contains **4,896 unique images**: 3,403 train, 993 validation, and 500 test. Duplicate images were kept within one split to avoid exact-image leakage.
- **Class imbalance:** Images containing NO-Safety Vest were less represented. Their paths were listed twice in the training manifest, increasing their exposure without creating new unique photographs. Other classes in those same images were repeated too. Validation and test splits were not oversampled.
- **Augmentation:** The training configuration explicitly sets horizontal flip and moderate HSV changes. The saved run settings also show Ultralytics defaults including mosaic, translation, and scaling. These transformations were applied during training, not the final validation/test evaluation.
- **Model:** Pretrained **YOLOv8s and YOLOv11s**, an object detector. Both strategies started independently from the same pretrained weights:
  - **Frozen backbone:** `freeze=10` freezes backbone modules 0–9 and trains the neck and detection head.
  - **Partial fine-tuning:** `freeze=8` keeps modules 0–7 frozen and additionally trains backbone modules 8–9, the neck, and the head. 

Both runs used 20 epochs, 640-pixel training images, batch size 16, AdamW, and the same training list and augmentation settings. The best checkpoint from each run was compared on validation images.

## Workflow / architecture

1. Curate and deduplicate the four-class dataset; keep train, validation, and test images separate.
2. Prepare the training-only oversampling list and augmentation settings.
3. Train the two YOLOv8s strategies independently.
4. Compare their best checkpoints on validation data and select the higher overall mAP50-95.
5. Evaluate the selected checkpoint once on the held-out test split.
6. Load the selected `.pt` weights locally and run webcam inference: frame → YOLO detector → labeled boxes and confidence scores.


## Results and evaluation


### YOLOv8s validation comparison

| Strategy | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen backbone | 0.708 | **0.759** | 0.727 | 0.763 | 0.456 |
| Partial fine-tuning | **0.753** | 0.758 | **0.752** | **0.782** | **0.468** |

The partial fine-tuning model was selected because its overall validation mAP50-95 was higher. This is not a win on every class: validation Hardhat recall was **0.676** for partial fine-tuning versus **0.723** for the frozen backbone. For NO-Safety Vest, partial fine-tuning improved precision from 0.491 to 0.541, while recall decreased from 0.724 to 0.707.

### Held-out test of the selected YOLOv8s model

| Precision | Recall | F1 | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.758 | 0.741 | 0.744 | 0.768 | 0.449 |

The test used 500 images and 1,478 annotated instances. Safety Vest had the highest test mAP50-95 (0.610). NO-Safety Vest was harder (0.341; recall 0.748), and NO-Hardhat recall was 0.654. The overall validation-to-test mAP50-95 difference was 0.019. These results describe the curated dataset; they do not guarantee equal performance on a new webcam scene.

### YOLO11s experiment

The YOLO11s experiment repeated the two freezing strategies with the same dataset and training settings:

| Strategy | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen backbone | 0.732 | **0.768** | 0.750 | 0.788 | 0.470 |
| Partial fine-tuning | **0.809** | 0.765 | **0.784** | **0.817** | **0.491** |

YOLO11s partial fine-tuning was selected because it achieved the higher validation mAP50-95. Its Hardhat validation recall also rose from 0.700 to 0.768. The selected partial model was then evaluated on the held-out test split:

| Model on test | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s partial | 0.758 | 0.741 | 0.744 | 0.768 | 0.449 |
| YOLO11s partial | **0.792** | **0.763** | **0.774** | **0.798** | **0.472** |

YOLO11s test Hardhat recall was **0.775**, compared with **0.728** for YOLOv8s. Its NO-Safety Vest precision/recall were **0.656 / 0.777**.

### Prediction examples and failure analysis

- **Successful example:** The local webcam demo detected NO-Safety Vest and NO-Hardhat.
- **Failure case:** A worn hardhat in the local webcam scene was detected at very low confidence, especially as viewpoint and position changed.


## Deployment and optimization

The working demo uses the selected PyTorch checkpoint `ppe_best.pt` with Ultralytics and OpenCV in VS Code. ONNX export is documented as an **deployment step**

## Technologies

Python, Google Colab, PyTorch, Ultralytics YOLOv8 and YOLO11, OpenCV, pandas, Matplotlib, and Google Drive for storing the dataset archive and training outputs.

## How to run

**Webcam demo (VS Code):** Install `ultralytics` and `opencv-python` in the selected Python environment. Download run the model training notebook  and put it in the VS Code working directory next to the webcam notebook. Run its cells using a **local** Python kernel. The notebook opens a desktop webcam window; press **q** in that window to stop. You can also add an image path instead of using the webcam.

## Suggested repository layout

```text
README.md
notebooks/
  Data_analysis.ipynb
  PPE_YOLOv8s_FineTuning.ipynb
  PPE_YOLOv11s_FineTuning.ipynb
  PPE_YOLOv_Local_Webcam_Inference.ipynb
workflow.png
far.jpeg
```

## Future improvements

- Add more annotated close-up hardhat images from the intended webcam setting, including different distances, angles, and lighting. Keep separate new captures for evaluation to check whether close-up detection improves.
- Compare different layer-freezing choices. Both current strategies trained the entire neck and head; the difference was how much of the backbone was unfrozen. A new experiment could freeze or unfreeze selected neck modules and compare the results with the current strategies.
- Try more than 20 epochs under the same dataset and training settings. Compare validation performanc

## SDAIA Academy GitHub repository

[**SDAIA Academy GitHub repository link.** 
](https://github.com/SDAIAAcademy)
