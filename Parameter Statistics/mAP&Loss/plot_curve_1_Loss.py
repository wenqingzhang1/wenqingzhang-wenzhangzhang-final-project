import re
import matplotlib.pyplot as plt
import os

# ================= Configuration Area =================
# Modify this to your actual saved log file path
LOG_FILE = 'training_log_2026-04-15_17-31-04.txt' 
# Image save path
OUTPUT_IMAGE = 'Loss_training_result.png'
# ===========================================

def parse_log(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Cannot find file {file_path}, please check if the path is correct.")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Dictionary to store data
    epoch_losses = {} # {epoch: [loss1, loss2, ...]}
    maps = []         # [(epoch, map_value)]
    
    # Regular expression matching
    
    # Match training Loss, e.g.: >> Train: [1][10/125] ... Loss 0.0001 (0.0132)
    loss_pattern = re.compile(r'>> Train: \[(\d+)\].*Loss ([\d\.]+)')
    
    # Match validation mAP, e.g.: >> sop: mAP 55.43
    # Note: validation usually happens at the end of each Epoch
    map_pattern = re.compile(r'>> sop: mAP ([\d\.]+)')

    current_map_epoch = 1

    for line in lines:
        # 1. Extract Loss
        loss_match = loss_pattern.search(line)
        if loss_match:
            epoch = int(loss_match.group(1)) # Let Epoch start from 1
            loss_val = float(loss_match.group(2))
            
            if epoch not in epoch_losses:
                epoch_losses[epoch] = []
            epoch_losses[epoch].append(loss_val)

        # 2. Extract mAP
        map_match = map_pattern.search(line)
        if map_match:
            map_val = float(map_match.group(1))
            maps.append((current_map_epoch, map_val))
            current_map_epoch += 1

    # Compute average Loss for each Epoch
    epochs = sorted(epoch_losses.keys())
    avg_losses = [sum(epoch_losses[e])/len(epoch_losses[e]) for e in epochs]
    
    # Organize mAP data (ensure x-axis alignment)
    map_epochs = [m[0] for m in maps]
    map_values = [m[1] for m in maps]

    return epochs, avg_losses, (map_epochs, map_values)

def plot_and_save(epochs, losses, map_data):
    map_epochs, map_values = map_data
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Set title and general style
    plt.title('SOP Dataset Training: Loss & mAP')
    plt.grid(True, linestyle='--', alpha=0.5)

    # === Plot left axis (Loss) ===
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    ax1.plot(epochs, losses, color=color, marker='o', linewidth=2, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # === Plot right axis (mAP) ===
    if map_values:
        ax2 = ax1.twinx()  # Instantiate second axis
        color = 'tab:blue'
        ax2.set_ylabel('Validation mAP (%)', color=color, fontweight='bold')
        ax2.plot(map_epochs, map_values, color=color, marker='s', linewidth=2, linestyle='--', label='Val mAP')
        ax2.tick_params(axis='y', labelcolor=color)
    else:
        print("Note: No mAP data found in the log, only the Loss curve will be plotted.")

    # Adjust layout to prevent label overlap
    fig.tight_layout()
    
    # Save image
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Success! The plot has been saved as: {OUTPUT_IMAGE}")
    # If you want to display in a window, uncomment the line below
    # plt.show()

if __name__ == "__main__":
    print("Parsing log...")
    ep, ls, mp = parse_log(LOG_FILE)
    
    if ep:
        print(f"Parsed training data for {len(ep)} Epochs.")
        plot_and_save(ep, ls, mp)
    else:
        print("No valid data parsed, please check the log file format.")