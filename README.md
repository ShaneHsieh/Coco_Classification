# Classification for coco dataset


Dowload coco dataset.

https://cocodataset.org/#download

1. 2017 Train images
2. 2017 Val images
3. 2017 Train/Val annotations
## Installation

Using conda to run this project

```bash
conda create --name coco_classification python=3.10
conda activate coco_classification
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pycocotools opencv-python pillow tqdm scikit-learn seaborn
pip install imgaug
```
    
Note:

The imgaug lib neet to fix code when using  NumPy 2.0.

Issue: https://github.com/aleju/imgaug/issues/859
## Run Locally

Clone the project

```bash
  git clone https://github.com/ShaneHsieh/Coco_Classification.git
```

Go to the project directory

```bash
  cd Coco_Classification
```

Prepare coco dataset for classification dataset

```bash
  python data_prepare.py --annotation_path Data/instances_train2017.json --image_dir Data/train2017 --output_dir cropped_dataset/train
  python data_prepare.py --annotation_path Data/instances_val2017.json --image_dir Data/val2017 --output_dir cropped_dataset/val
```

Start train model

```bash
  #using default variable
  python train.py
  #or
  python train.py --data_dir cropped_dataset --num_classes 4 --batch_size 32 --num_epochs 10 --learning_rate 0.0001 --device 0
```
