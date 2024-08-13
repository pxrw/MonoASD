# 🍉MonoASD: Align and Select Distillation for Monocular 3D Object Detection

## 🍎Introduction

---

This is the repo of MonoASD: Align and Select Distillation for Monocular 3D Object Detection.

![avar](E:\PHD\论文\Distillation-AAAI\MonoASD\img\framework.jpg)

---

## 🍒Abstract

The task of 3D detection has been a significant focus in computer vision research, with monocular 3D detection gaining considerable attention due to its potential to accurately determine the position of an object in 3D space from a single image at a low cost. Knowledge distillation techniques are effective for achieving inter-modal knowledge transfer. Current knowledge distillation methods for monocular 3D detection typically train a teacher network using 2D depth data and transfer knowledge from the teacher network to the student network. However, this distillation is often complex and inefficient, primarily due to the uncertainty in achieving alignment between the distilled features, which can introduce erroneous distillation information. To address these issues and enhance the effectiveness of information transfer during the distillation, we propose the AlignSelect distillation method called MonoASD. Our proposed model forces the student model features to align with the teacher model features during distillation in training, selects important features from the long sequence to calculate the distillation loss, and completes the information transfer efficiently. Unlike other distillation methods, our approach uses the aligned and selected features as inputs for subsequent processing, forming an in-the-loop distillation. Extensive experiments conducted on the KITTI 3D object detection benchmark validate the effectiveness of our proposed method. Our method achieves state-of-the-art (SOTA) results without introducing any additional computational cost during inference.

---

## 🍓 Getting Start

### Dataset Preparation

*   Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows:

    ```
    this repo
    ├── data
    │   │── KITTI3D
    |   │   │── training
    |   │   │   ├──calib & label_2 & image_2 & depth_dense
    |   │   │── testing
    |   │   │   ├──calib & image_2
    ├── config
    ├── ...
    ```

*   You can also choose to link your KITTI dataset path by

    ```
      KITTI_DATA_PATH=~/data/kitti_object
      ln -s $KITTI_DATA_PATH ./data/KITTI3D
    ```

*   To ease the usage,  the pre-generated dense depth files at: [Google Drive](https://drive.google.com/file/d/1mlHtG8ZXLfjm0lSpUOXHulGF9fsthRtM/view?usp=sharing) 

---

## 🍇Training & Testing

### Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_val.py --config configs/monoasd.yaml
```

### Test and evaluate 

```
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config configs/monoasd.yaml -e
```

