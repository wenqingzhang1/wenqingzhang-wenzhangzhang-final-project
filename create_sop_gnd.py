import os
import pickle
import numpy as np

# Path configuration
DATA_ROOT = 'data'
TRAIN_PKL = os.path.join(DATA_ROOT, 'train', 'sop', 'sop.pkl')
TEST_DIR = os.path.join(DATA_ROOT, 'test', 'sop')
OUTPUT_GND = os.path.join(TEST_DIR, 'gnd_sop.pkl')

def create_gnd():
    print(f"Reading: {TRAIN_PKL}")
    with open(TRAIN_PKL, 'rb') as f:
        # Read the previously generated validation set (val) data
        data = pickle.load(f)['val']
    
    cids = data['cids']      # All image paths
    clusters = data['cluster'] # Class ID corresponding to each image
    qidxs = data['qidxs']    # Indices of query images
    pidxs = data['pidxs']    # Indices of positive samples; not actually used here, we recalculate them

    # 1. Create test directory structure
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)
        # The SOP test set can directly link to the training ims directory.
        # To save space, we can use a symbolic link or reuse the path directly.
        # When modifying testdataset.py, we will point it to the correct location.
        # Here we only create the folder first to store the gnd file.

    # 2. Build the Ground Truth (gnd) list
    # Structure: gnd[i]['ok'] = [indices of all database images in the same class as the query image]
    print("Building validation set Ground Truth...")
    gnd = []
    
    # Convert to numpy for easier calculation
    clusters_np = np.array(clusters)
    
    for i, q_idx in enumerate(qidxs):
        q_cluster = clusters[q_idx]
        
        # Find all image indices in the database that belong to the same class, excluding the query image itself
        # Note: In SOP, all images in the validation set with the same class except itself are treated as correct answers (ok)
        ok_indices = np.where(clusters_np == q_cluster)[0]
        ok_indices = ok_indices[ok_indices != q_idx] # Remove itself
        
        gnd.append({
            'ok': ok_indices,
            'junk': np.array([]) # SOP has no ambiguous (junk) images, so leave it empty
        })

        if (i+1) % 100 == 0:
            print(f"\rProcessing progress: {i+1}/{len(qidxs)}", end='')

    # 3. Save the file
    # We also need to save imlist (all database images) and qimlist (query images)
    # so that testdataset.py knows which files to read
    db_structure = {
        'imlist': cids,  # Database image list; actually all images in the validation set
        'qimlist': [cids[i] for i in qidxs], # Query image list
        'gnd': gnd
    }

    with open(OUTPUT_GND, 'wb') as f:
        pickle.dump(db_structure, f)
    
    print(f"\n\nSuccessfully generated: {OUTPUT_GND}")
    print("Preparation completed!")

if __name__ == '__main__':
    create_gnd()