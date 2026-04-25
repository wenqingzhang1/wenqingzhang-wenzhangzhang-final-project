import os
import pickle
import random

# === Configuration Paths ===
# Your images should be in this location: data/train/sop/ims
# The index file we generate will be placed at: data/train/sop/sop.pkl
ROOT_DIR = 'data/train/sop'
IMS_DIR = os.path.join(ROOT_DIR, 'ims')
OUTPUT_PKL = os.path.join(ROOT_DIR, 'sop.pkl')

def create_db():
    # Check whether the image directory exists
    if not os.path.exists(IMS_DIR):
        print(f"Error: Cannot find image directory {IMS_DIR}")
        print("Please check whether Step 1 is correct: the folder structure should be data/train/sop/ims/bicycle_final/...")
        return

    print(f"Scanning directory: {IMS_DIR} ...")
    
    # 1. Scan all folders
    cids = []      # Store relative image paths, for example: bicycle_final/123.jpg
    clusters = []  # Store class IDs, for example: 0, 0, 1, 2...
    
    # Get all subfolders under ims, ignoring files such as LICENSE, .txt, etc.
    classes = [d for d in os.listdir(IMS_DIR) if os.path.isdir(os.path.join(IMS_DIR, d))]
    classes.sort() # Sort to ensure the IDs are the same every time the script runs
    
    # Assign a numeric ID to each folder
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Found {len(classes)} product categories. Starting to build the index...")

    for cls_name in classes:
        cls_dir = os.path.join(IMS_DIR, cls_name)
        # Get all images under this folder
        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img in imgs:
            # Key point: this stores the path relative to the ims folder
            rel_path = os.path.join(cls_name, img)
            # For compatibility across different systems, replace backslashes with forward slashes
            rel_path = rel_path.replace('\\', '/')
            
            cids.append(rel_path)
            clusters.append(class_to_idx[cls_name])

    print(f"Scanning completed: found {len(cids)} images in total.")

    # 2. Create the database dictionary
    # For simplicity, use 90% of the images for training and 10% for validation
    num_samples = len(cids)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices) # Shuffle the order
    
    split_idx = int(num_samples * 0.9)
    train_idx = all_indices[:split_idx]
    val_idx = all_indices[split_idx:]
    
    db = {}
    
    # Training set structure
    db['train'] = {
        'cids': cids,
        'cluster': clusters,
        'qidxs': train_idx, # During training, these images can be used as Query images
        'pidxs': train_idx  # They can also be used as Positive images
    }
    
    # Validation set structure
    db['val'] = {
        'cids': cids,
        'cluster': clusters,
        'qidxs': val_idx,
        'pidxs': val_idx
    }
    
    # 3. Save the file
    print(f"Saving to {OUTPUT_PKL} ...")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(db, f)
    
    print("All done! Step 2 completed.")

if __name__ == '__main__':
    create_db()