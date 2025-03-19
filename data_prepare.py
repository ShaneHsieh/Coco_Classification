import argparse
from pycocotools.coco import COCO
import os
import cv2
from tqdm import tqdm

target_classes = {
    3: 'Car',
    62: 'Chair',
    44: 'bottle',
    84: 'book'
}

def process_dataset(annotation_path, image_dir, output_dir, max_images_per_class=None):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        coco = COCO(annotation_path)
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return
    
    for cat_id, class_name in target_classes.items():
        # get image that containing this category ID
        image_ids = coco.getImgIds(catIds=[cat_id])
        
        if max_images_per_class is not None:
            image_ids = image_ids[:max_images_per_class]
            
        save_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        #process all image
        for img_id in tqdm(image_ids, desc=f"Processing {class_name}"):
            #load image info
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])

            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            # get all label
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[cat_id], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            
            for ann in anns:
                x, y, w, h = map(int, ann['bbox'])
                cropped = img[y:y+h, x:x+w]
                
                # filter too small object
                if cropped.shape[0] < 32 or cropped.shape[1] < 32:
                    continue
                    
                out_name = f"{img_id}_{ann['id']}.jpg"
                cv2.imwrite(os.path.join(save_dir, out_name), cropped)

def parse_args():
    parser = argparse.ArgumentParser(description='Process COCO dataset and crop objects')
    parser.add_argument('--annotation_path', type=str, required=True,
                       help='Path to COCO annotation JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing the images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for cropped images')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process per class (optional)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    process_dataset(
        annotation_path=args.annotation_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_images_per_class=args.max_images
    )