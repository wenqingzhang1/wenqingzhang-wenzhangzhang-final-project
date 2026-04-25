import re
import matplotlib.pyplot as plt
import os

# ================= Configuration Area =================
# Modify this to your actual saved log file path
LOG_FILE = 'training_log_2026-04-15_17-31-04.txt'
# Image save path
OUTPUT_IMAGE = 'mAP_training_result.png'
# ===========================================


def parse_log(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Cannot find file {file_path}, please check if the path is correct.")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Dictionary to store data
    epoch_losses = {}   # {epoch: [loss1, loss2, ...]}
    maps = []           # [(epoch, map_value)]

    # ================= Regular expression matching =================
    # Match training Loss, for example:
    # >> Train: [1][10/125]	Time ...	Loss 0.0000 (0.0108)
    loss_pattern = re.compile(r'>> Train: \[(\d+)\].*?Loss ([\d\.]+)')

    # Match validation mAP, for example:
    # >> mAP:         25.52%
    map_pattern = re.compile(r'>> mAP:\s*([\d\.]+)%')
    # ================================================

    current_map_epoch = 1

    for line in lines:
        # 1. Extract Loss
        loss_match = loss_pattern.search(line)
        if loss_match:
            # In this log, epoch already starts from 1, no need to +1
            epoch = int(loss_match.group(1))
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

    # If no loss is extracted, return empty
    if not epoch_losses:
        return None, None, None

    # Compute average Loss for each Epoch
    epochs = sorted(epoch_losses.keys())
    avg_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]

    # Organize mAP data
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

    # Annotate loss values on the curve
    for x, y in zip(epochs, losses):
        ax1.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    # === Plot right axis (mAP) ===
    if map_values:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation mAP (%)', color=color, fontweight='bold')
        ax2.plot(map_epochs, map_values, color=color, marker='s', linewidth=2, linestyle='--', label='Val mAP')
        ax2.tick_params(axis='y', labelcolor=color)

        # Annotate mAP values on the curve
        for x, y in zip(map_epochs, map_values):
            ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    else:
        print("Note: No mAP data found in the log, only the Loss curve will be plotted.")

    # Set x-axis ticks as integer epochs
    ax1.set_xticks(epochs)

    # Merge legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if map_values:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    else:
        ax1.legend(loc='best')

    # Adjust layout to prevent label overlap
    fig.tight_layout()

    # Save image
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Success! The plot has been saved as: {OUTPUT_IMAGE}")

    # If you want to display in a window, uncomment the line below
    # plt.show()


if __name__ == "__main__":
    print("Parsing log...")
    ep, ls, mp = parse_log(LOG_FILE)

    if ep:
        print(f"Parsed training data for {len(ep)} Epochs.")
        if mp[1]:
            print(f"Parsed mAP data for {len(mp[1])} Epochs.")
        else:
            print("No mAP data parsed.")
        plot_and_save(ep, ls, mp)
    else:
        print("No valid data parsed, please check the log file format.")