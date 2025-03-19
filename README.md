# Classification for coco dataset


Dowload coco dataset.

https://cocodataset.org/#download

1. 2017 Train images
2. 2017 Val images
3. 2017 Train/Val annotations
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
