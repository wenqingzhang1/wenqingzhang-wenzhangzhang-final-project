import os
import time
import torch
import numpy as np
import glob  # <--- Added: used to automatically search files through deeply nested folders
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.general import get_data_root

def compute_real_recall(ranks, gnd, kappas=[1, 10, 100]):
    """
    Compute the real academic Recall@K.
    Logic: after excluding the query image itself, if Top-K contains at least one image of the same product, it is counted as a hit.
    """
    nq = ranks.shape[1]
    recalls = {k: 0.0 for k in kappas}
    
    for q in range(nq):
        # Get correct matching indices (ok) and ignored indices (junk, usually the query image itself)
        junk_indices = set(gnd[q].get('junk', []))
        ok_indices = set(gnd[q]['ok'])
        
        # Core: remove the query image itself from the retrieval results
        valid_ranks = [idx for idx in ranks[:, q] if idx not in junk_indices]
        
        # Count Recall@K
        for k in kappas:
            # If any of the first k valid_ranks belongs to ok_indices, it is counted as a hit
            if any(idx in ok_indices for idx in valid_ranks[:k]):
                recalls[k] += 1
                
    # Compute percentage
    return {k: (v / nq) * 100.0 for k, v in recalls.items()}

def main():
    print("\n[*] 🚀 Starting offline authoritative evaluation (Offline Evaluation)")
    
    # 1. Automatically search for the weight file path through deeply nested long-name subfolders
    base_dir = './exp_sop_resnet50_v4'
    # Use ** to match all subdirectories and search for model_best.pth.tar
    search_pattern = os.path.join(base_dir, '**', 'model_best.pth.tar')
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        raise FileNotFoundError(f"model_best.pth.tar was not found in {base_dir} or its subdirectories!\nPlease check whether the weight file was successfully generated in this directory.")
    
    checkpoint_path = found_files[0]  # Automatically use the first found weight file
    print(f"[*] Successfully located the weight file automatically: \n    {checkpoint_path}")

    # 2. Load model architecture and weights
    print(f"[*] Loading model weights...")
    checkpoint = torch.load(checkpoint_path)
    net_params = checkpoint['meta']
    net = init_network(net_params)
    net.load_state_dict(checkpoint['state_dict'])
    
    # Move to GPU and set to evaluation mode
    net.cuda()
    net.eval()

    # 3. Set preprocessing pipeline
    normalize = transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # To prevent OOM on 8GB VRAM, still use the training resolution of 224 for feature extraction
    image_size = 224

    # 4. Load SOP test set configuration
    print("[*] Loading SOP test set information...")
    cfg = configdataset('sop', os.path.join(get_data_root(), 'test'))
    images = [cfg['im_fname'](cfg, i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg, i) for i in range(cfg['nq'])]
    
    # Get the ground truth dictionary of the test set
    gnd = cfg['gnd']

    # 5. Extract feature vectors independently from the original DataLoader
    print(f"[*] Extracting features from {len(images)} database images... (please wait patiently)")
    start_time = time.time()
    with torch.no_grad():
        vecs = extract_vectors(net, images, image_size, transform).numpy()
        print(f"[*] Extracting features from {len(qimages)} query images...")
        qvecs = extract_vectors(net, qimages, image_size, transform).numpy()
    
    print(f"[*] Feature extraction completed! Time used: {time.time() - start_time:.2f} seconds")

    # 6. Compute cosine similarity and global ranking
    print("[*] Computing cosine similarity and global ranking...")
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)

    # 7. Compute real Recall@K
    print("\n" + "="*50)
    print(" 🏆 Real Recall Evaluation Results on the SOP Dataset (Golden Standard)")
    print("="*50)
    
    recalls = compute_real_recall(ranks, gnd, kappas=[1, 10, 100])
    
    print(f" >> \033[92mRecall@1   : {recalls[1]:.2f}%\033[0m")
    print(f" >> \033[92mRecall@10  : {recalls[10]:.2f}%\033[0m")
    print(f" >> \033[92mRecall@100 : {recalls[100]:.2f}%\033[0m")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()