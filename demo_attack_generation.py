#!/usr/bin/env python3
"""
Demonstration script for AIDM attack generation using digital-twin-dataset sample data.
Shows how to generate realistic attacks based on real power system measurements.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import logging
from pathlib import Path

from attacks import AttackGenerator
from data_loader import load_sample_data
from visualize_attacks import create_attack_visualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate attack generation with sample data."""
    
    print("🚀 AIDM Attack Generation Demo")
    print("=" * 50)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📁 Dataset path: {config['data']['dataset_path']}")
    print(f"📊 Small data mode: {config['compute']['small_data_mode']}")
    
    # Initialize attack generator
    generator = AttackGenerator(config)
    generator.initialize_power_model()
    
    # Load sample data from digital-twin-dataset
    try:
        print("\n📥 Loading sample data...")
        data = load_sample_data(
            dataset_path=config['data']['dataset_path'],
            small_data_mode=config['compute']['small_data_mode'],
            synthetic_fallback=True,
            use_api=False  # Use sample data only
        )
        
        # Extract measurements
        if 'magnitude' in data and not data['magnitude'].empty:
            measurements = data['magnitude'].values
            timestamps = data['magnitude'].index
            data_type = "magnitude"
        elif 'phasor' in data and not data['phasor'].empty:
            measurements = data['phasor'].values
            timestamps = data['phasor'].index
            data_type = "phasor"
        else:
            # Use any available data
            available_keys = [k for k, v in data.items() if not v.empty]
            if available_keys:
                key = available_keys[0]
                measurements = data[key].values
                timestamps = data[key].index
                data_type = key
            else:
                raise ValueError("No valid data found")
        
        print(f"✅ Loaded {data_type} data: {measurements.shape}")
        print(f"📅 Time range: {timestamps[0]} to {timestamps[-1]}")
        
    except Exception as e:
        print(f"❌ Failed to load sample data: {e}")
        return
    
    # Generate attack dataset
    print("\n⚔️ Generating attacks...")
    
    attack_types = ['fdia', 'temporal_stealth', 'replay']
    attack_data = generator.generate_attack_dataset(
        measurements=measurements,
        timestamps=timestamps,
        attack_types=attack_types,
        attack_ratio=0.3  # 30% attacks, 70% clean
    )
    
    # Print results
    clean_samples = np.sum(attack_data['labels'] == 0)
    attack_samples = np.sum(attack_data['labels'] == 1)
    
    print(f"\n📊 Dataset Generated:")
    print(f"   Total samples: {len(attack_data['labels'])}")
    print(f"   Clean samples: {clean_samples} ({clean_samples/len(attack_data['labels'])*100:.1f}%)")
    print(f"   Attack samples: {attack_samples} ({attack_samples/len(attack_data['labels'])*100:.1f}%)")
    
    # Show attack type breakdown
    attack_mask = attack_data['labels'] == 1
    if np.any(attack_mask):
        attack_type_counts = pd.Series(attack_data['attack_types'][attack_mask]).value_counts()
        print(f"\n🎯 Attack Type Breakdown:")
        for attack_type, count in attack_type_counts.items():
            print(f"   {attack_type}: {count} samples")
    
    # Save the dataset
    print(f"\n💾 Saving dataset...")
    output_dir = Path(config['data']['output_path']) / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator.save_attack_dataset(attack_data, config['data']['output_path'], 'demo_sample_attacks')
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    try:
        visualizer = create_attack_visualizer(output_dir="../outputs/reports")
        
        # Create overview plot
        overview_fig = visualizer.plot_attack_overview(attack_data)
        overview_fig.write_html("outputs/reports/demo_attack_overview.html")
        
        # Create time series plot
        timeseries_fig = visualizer.plot_attack_timeseries(
            attack_data, 
            measurement_indices=list(range(min(3, measurements.shape[1]))),
            time_window=(0, min(500, len(measurements)))
        )
        timeseries_fig.write_html("outputs/reports/demo_attack_timeseries.html")
        
        print("✅ Visualizations saved to outputs/reports/")
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    # Quick analysis
    print(f"\n🔍 Quick Analysis:")
    
    # Attack magnitudes
    clean_data = attack_data['clean'][attack_data['labels'] == 0]
    attack_data_vals = attack_data['clean'][attack_data['labels'] == 1]
    
    if len(attack_data_vals) > 0 and len(clean_data) > 0:
        # Calculate attack magnitudes
        min_len = min(len(clean_data), len(attack_data_vals))
        attack_magnitudes = np.linalg.norm(
            attack_data_vals[:min_len] - clean_data[:min_len], axis=1
        )
        
        print(f"   Average attack magnitude: {np.mean(attack_magnitudes):.4f}")
        print(f"   Max attack magnitude: {np.max(attack_magnitudes):.4f}")
        print(f"   Min attack magnitude: {np.min(attack_magnitudes):.4f}")
    
    # Data statistics
    print(f"   Clean data mean: {np.mean(clean_data):.4f}")
    print(f"   Clean data std: {np.std(clean_data):.4f}")
    if len(attack_data_vals) > 0:
        print(f"   Attack data mean: {np.mean(attack_data_vals):.4f}")
        print(f"   Attack data std: {np.std(attack_data_vals):.4f}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Files saved in: {config['data']['output_path']}/experiments/")
    print(f"📊 Visualizations in: outputs/reports/")
    print(f"\n💡 Next steps:")
    print(f"   1. Use 'demo_sample_attacks_attacks.npz' for IDS training")
    print(f"   2. Open HTML files in browser for interactive analysis")
    print(f"   3. Run: python src/train_ids.py --model all")

if __name__ == "__main__":
    main()
