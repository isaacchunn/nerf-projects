"""
Extract all metrics from trained models (TensorBoard + Image Quality)

Usage:
    # Extract TensorBoard metrics from all scenes:
    python extract_metrics.py
    
    # Or from single scene:
    python extract_metrics.py ckpt/lego_syn_1015_054314
"""

import argparse
from pathlib import Path
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_dir', type=str, nargs='?', help='Single checkpoint directory (optional)')
args = parser.parse_args()

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard not found. Install with: pip install tensorboard")
    sys.exit(1)

# Find checkpoints
if args.checkpoint_dir:
    ckpt_path = Path(args.checkpoint_dir)
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found")
        sys.exit(1)
    ckpt_dirs = [ckpt_path]
else:
    ckpt_dir = Path("ckpt")
    if not ckpt_dir.exists():
        print("Error: ckpt/ directory not found")
        sys.exit(1)
    ckpt_dirs = [d for d in ckpt_dir.iterdir() 
                 if d.is_dir() and list(d.glob('events.out.tfevents.*'))]

if not ckpt_dirs:
    print("No checkpoints with TensorBoard logs found")
    sys.exit(1)

print(f"Found {len(ckpt_dirs)} checkpoint(s)")
print("="*80)

# Extract metrics from each
all_data = []

for ckpt_path in sorted(ckpt_dirs):
    scene_name = ckpt_path.name
    print(f"\n{scene_name}")
    
    try:
        # Load TensorBoard events
        ea = event_accumulator.EventAccumulator(str(ckpt_path))
        ea.Reload()
        scalar_tags = ea.Tags()['scalars']
        
        # Extract ALL scalar metrics (final values)
        metrics = {'Scene': scene_name}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            if events:
                # Clean tag name for CSV column
                clean_tag = tag.replace('/', '_').replace('__', '_')
                metrics[clean_tag] = events[-1].value
        
        # Try to read image metrics if they exist
        test_renders = ckpt_path / "test_renders"
        if test_renders.exists():
            metrics_summary = test_renders / "metrics_summary.txt"
            if metrics_summary.exists():
                with open(metrics_summary, 'r') as f:
                    for line in f:
                        if 'PSNR:' in line:
                            metrics['PSNR_from_renders'] = float(line.split(':')[1].strip().split()[0])
                        elif 'SSIM:' in line:
                            metrics['SSIM'] = float(line.split(':')[1].strip())
                        elif 'LPIPS:' in line:
                            metrics['LPIPS'] = float(line.split(':')[1].strip())
        
        all_data.append(metrics)
        print(f"  ✓ Extracted {len(metrics)-1} metrics")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")

if not all_data:
    print("\nNo data extracted")
    sys.exit(1)

# Create output directory
output_dir = Path("metrics_summary")
output_dir.mkdir(exist_ok=True)

# Get all unique column names
all_columns = set()
for row in all_data:
    all_columns.update(row.keys())

# Order columns: Scene first, then key metrics, then alphabetical
key_metrics = ['Scene', 'test_psnr', 'PSNR_from_renders', 'SSIM', 'LPIPS',
               'metrics_FDR', 'metrics_MCQ', 'metrics_num_floaters', 
               'metrics_peak_gpu_gb', 'epoch_id']
ordered_columns = [c for c in key_metrics if c in all_columns]
ordered_columns += sorted([c for c in all_columns if c not in ordered_columns])

# Write single comprehensive CSV
output_csv = output_dir / "all_metrics.csv"
with open(output_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=ordered_columns)
    writer.writeheader()
    for row in all_data:
        writer.writerow(row)

print("\n" + "="*80)
print(f"✓ Metrics saved to: {output_csv}")
print(f"  - {len(all_data)} scenes")
print(f"  - {len(ordered_columns)} metrics per scene")

# Print summary table (key metrics only)
print("\n" + "="*80)
print("KEY METRICS SUMMARY")
print("="*80)

display_cols = [c for c in ['Scene', 'test_psnr', 'SSIM', 'LPIPS', 'metrics_FDR', 
                             'metrics_num_floaters'] if c in ordered_columns]

# Header
for col in display_cols:
    print(f"{col[:20]:>20}", end="  ")
print()
print("-"*80)

# Data
for row in all_data:
    for col in display_cols:
        val = row.get(col, '')
        if isinstance(val, float):
            print(f"{val:20.4f}", end="  ")
        else:
            print(f"{str(val):>20}", end="  ")
    print()

print("="*80)

# Print instructions
print("\nTo get SSIM/LPIPS (if missing):")
print("  1. Generate test renders:")
print("     python render_imgs.py ckpt/<scene>/ckpt.npz data/nerf_synthetic/<scene>")
print("  2. Compute metrics:")
print("     python calc_metrics.py ckpt/<scene>/test_renders --dataset_type syn --data_dir data/nerf_synthetic/<scene>")
print("  3. Re-run this script to include SSIM/LPIPS")

