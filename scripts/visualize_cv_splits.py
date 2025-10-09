#!/usr/bin/env python3
"""
Visualize how train/dev/test splits are used in RAG optimization.
Creates a diagram showing data flow through the 5-fold CV process.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('Cross-Validation Split Usage in RAG Hyperparameter Optimization',
             fontsize=16, fontweight='bold', y=0.98)

# Colors
color_train = '#3498db'  # Blue
color_dev = '#2ecc71'    # Green
color_test = '#e74c3c'   # Red
color_unused = '#95a5a6'  # Gray
color_final = '#9b59b6'  # Purple

# ==========================================
# SUBPLOT 1: Before (Only Dev Used)
# ==========================================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
ax1.set_title('BEFORE: Only Dev Set Used for Optimization (Inefficient)',
              fontsize=14, fontweight='bold', pad=20)

fold_height = 0.8
fold_y_start = 5

# Draw folds 0-3
for i in range(4):
    y = fold_y_start - i * 1.2

    # Fold label
    ax1.text(-0.5, y + fold_height / 2, f'Fold {i}\n(optim)',
             ha='right', va='center', fontsize=10, fontweight='bold')

    # Train (60%) - UNUSED
    train_box = FancyBboxPatch((0, y), 6, fold_height,
                               boxstyle="round,pad=0.05",
                               facecolor=color_unused, edgecolor='black', linewidth=2, alpha=0.3)
    ax1.add_patch(train_box)
    ax1.text(3, y + fold_height / 2, 'Train 60%\n❌ UNUSED',
             ha='center', va='center', fontsize=9, fontweight='bold')

    # Dev (20%) - USED
    dev_box = FancyBboxPatch((6, y), 2, fold_height,
                             boxstyle="round,pad=0.05",
                             facecolor=color_dev, edgecolor='black', linewidth=2)
    ax1.add_patch(dev_box)
    ax1.text(7, y + fold_height / 2, 'Dev 20%\n✅ USED',
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Test (20%) - UNUSED
    test_box = FancyBboxPatch((8, y), 2, fold_height,
                              boxstyle="round,pad=0.05",
                              facecolor=color_unused, edgecolor='black', linewidth=2, alpha=0.3)
    ax1.add_patch(test_box)
    ax1.text(9, y + fold_height / 2, 'Test 20%\n❌ UNUSED',
             ha='center', va='center', fontsize=9, fontweight='bold')

# Draw Fold 4 (final test)
y = fold_y_start - 4 * 1.2
ax1.text(-0.5, y + fold_height / 2, 'Fold 4\n(final)',
         ha='right', va='center', fontsize=10, fontweight='bold')

final_test_box = FancyBboxPatch((0, y), 10, fold_height,
                                boxstyle="round,pad=0.05",
                                facecolor=color_final, edgecolor='black', linewidth=2)
ax1.add_patch(final_test_box)
ax1.text(5, y + fold_height / 2, 'Final Test 100% (20% of total dataset)\n✅ USED ONCE',
         ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Add summary text
summary_y = y - 1
ax1.text(5, summary_y,
         '⚠️ Problem: Only 20% of data used per fold\n'
         'Total evaluations per config: 4 × 20% = 0.8× dataset size\n'
         'Result: High variance, wasted data',
         ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#ffe6e6', edgecolor='red', linewidth=2, alpha=0.7))

# ==========================================
# SUBPLOT 2: After (Train+Dev Used)
# ==========================================
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('AFTER: Train+Dev Merged for Optimization (Efficient)',
              fontsize=14, fontweight='bold', pad=20)

# Draw folds 0-3
for i in range(4):
    y = fold_y_start - i * 1.2

    # Fold label
    ax2.text(-0.5, y + fold_height / 2, f'Fold {i}\n(optim)',
             ha='right', va='center', fontsize=10, fontweight='bold')

    # Train+Dev (80%) - USED
    traindev_box = FancyBboxPatch((0, y), 8, fold_height,
                                  boxstyle="round,pad=0.05",
                                  facecolor=color_train, edgecolor='black', linewidth=2)
    ax2.add_patch(traindev_box)
    ax2.text(4, y + fold_height / 2, 'Train+Dev 80% (merged)\n✅ USED FOR EVALUATION',
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Test (20%) - UNUSED
    test_box = FancyBboxPatch((8, y), 2, fold_height,
                              boxstyle="round,pad=0.05",
                              facecolor=color_unused, edgecolor='black', linewidth=2, alpha=0.3)
    ax2.add_patch(test_box)
    ax2.text(9, y + fold_height / 2, 'Test 20%\n❌ UNUSED',
             ha='center', va='center', fontsize=9, fontweight='bold')

# Draw Fold 4 (final test)
y = fold_y_start - 4 * 1.2
ax2.text(-0.5, y + fold_height / 2, 'Fold 4\n(final)',
         ha='right', va='center', fontsize=10, fontweight='bold')

final_test_box = FancyBboxPatch((0, y), 10, fold_height,
                                boxstyle="round,pad=0.05",
                                facecolor=color_final, edgecolor='black', linewidth=2)
ax2.add_patch(final_test_box)
ax2.text(5, y + fold_height / 2, 'Final Test 100% (20% of total dataset)\n✅ USED ONCE',
         ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Add summary text
summary_y = y - 1
ax2.text(5, summary_y,
         '✅ Solution: 80% of data used per fold (4× improvement)\n'
         'Total evaluations per config: 4 × 80% = 3.2× dataset size\n'
         'Result: Lower variance, all data utilized, more reliable optimization',
         ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#e6ffe6', edgecolor='green', linewidth=2, alpha=0.7))

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=color_train, edgecolor='black',
                   label='Train+Dev (Used for evaluation)'),
    mpatches.Patch(facecolor=color_dev, edgecolor='black',
                   label='Dev only (Old approach)'),
    mpatches.Patch(facecolor=color_purple, edgecolor='black',
                   label='Final Test (Unbiased)'),
    mpatches.Patch(facecolor=color_unused, edgecolor='black',
                   alpha=0.3, label='Unused (Held out)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=11, frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig('/home/spiros/Desktop/Thesis/benchmarks/cv_split_usage_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Diagram saved to: benchmarks/cv_split_usage_diagram.png")
plt.show()
