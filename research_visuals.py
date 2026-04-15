import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

def create_comparison_chart():
    
    
    csv_file = 'comparative_evaluation_table.csv'
    
    # 1. Check if the file exists
    if not os.path.exists(csv_file):
        print(f" Error: Could not find '{csv_file}'. Make sure it is in the same folder!")
        return

    print(f" Found '{csv_file}', loading data...")
    df = pd.read_csv(csv_file)

    # 2. Identify the 'Model' column and set it as the index (X-axis)
    model_col = 'Model' if 'Model' in df.columns else df.columns[0]
    df.set_index(model_col, inplace=True)

    # 3. Define the exact metrics we want to graph (Matching the CSV EXACTLY)
    desired_metrics = [
        'Accuracy',
        'Precision (Fraud)',
        'Recall (Fraud)',
        'F1-Score (Fraud)',
        'ROC-AUC',
        'PR-AUC',
        'MCC'
    ]
    
    # Check which of these columns actually exist in your CSV to prevent crashing
    actual_columns = [col for col in desired_metrics if col in df.columns]
    
    if not actual_columns:
        print(" Error: Could not find the metric columns in your CSV.")
        return

    # 4. Draw Individual Graphs for each Metric
    print(" Drawing the individual bar charts...")
    
    # Define a beautiful color palette for the different metrics
    metric_colors = {
        'Accuracy': '#3498db',         # Blue
        'Precision (Fraud)': '#9b59b6',# Purple
        'Recall (Fraud)': '#2ecc71',   # Green
        'F1-Score (Fraud)': '#e74c3c', # Red
        'ROC-AUC': '#f39c12',          # Orange
        'PR-AUC': '#8B4513',           # Teal
        'MCC': "#320453"               # Dark purple
    }

    # Define short labels to prevent x-axis collision on long model names
    short_names = {
        'Support Vector Machine (LinearSVC)': 'SVM',
        'K-Nearest Neighbors': 'KNN',
        'Decision Tree': 'Decision Tree',
        'Random Forest': 'Random Forest',
        'XGBoost': 'XGBoost'
    }

    # Loop through each metric and create its own chart
    for metric in actual_columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get the color for this metric, or default to teal
        color = metric_colors.get(metric, '#1abc9c')
        
        df[metric].plot(kind='bar', ax=ax, color=color, width=0.5)

        # Replace long model names with compact labels for the x-axis
        short_index = [short_names.get(label, label) for label in df.index]
        ax.set_xticklabels(short_index, rotation=30, ha='right', fontsize=12)

        # Clean up the metric name for beautiful titles and filenames
        clean_metric = metric.replace(" (Fraud)", "")

        # Use a shorter title so it stays inside the chart
        plt.title(f'{clean_metric} Comparison Across Models', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Machine Learning Models', fontsize=14, fontweight='bold')

        if metric == 'Accuracy':
            plt.ylabel('Accuracy Score (0.0 to 1.0)', fontsize=14, fontweight='bold')
            plt.ylim(0, 1.2)
        else:
            plt.ylabel(f'{clean_metric} Score (0.0 to 1.0)', fontsize=14, fontweight='bold')
            plt.ylim(0, 1.25)

        # 6.  Print the exact numbers on top of every single bar!
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                if metric == 'Accuracy':
                    display_height = np.floor(height * 1000) / 1000
                else:
                    display_height = height
                label = f"{display_height:.3f}"
                ax.annotate(label, 
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            xytext=(0, 10), # Push the text slightly above the bar
                            textcoords='offset points',
                            fontsize=12, 
                            fontweight='bold')

        # Add a faint grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 7. Save a high-resolution image for the paper
        safe_filename = clean_metric.replace(" ", "_").replace("-", "_")
        save_path = f'{safe_filename}_chart.png'
        plt.savefig(save_path, dpi=300) 
        print(f" Success! High-resolution graph saved as '{save_path}'")

def generate_confusion_matrices():
    
    json_file = 'model_performance_results.json'
    
    if not os.path.exists(json_file):
        print(f" Error: Could not find '{json_file}'. Please run your training scripts first.")
        return

    print(f"\n Found '{json_file}', generating confusion matrices...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    models = list(data.keys())
    num_models = len(models)
    
    if num_models == 0:
        print(" No models found in the JSON file.")
        return

    # Create a figure with subplots side-by-side
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    
    # Handle the case where there is only one model
    if num_models == 1:
        axes = [axes]
        
    for ax, model_name in zip(axes, models):
        cm = np.array(data[model_name]["confusion_matrix"])
        
        # Create a beautiful heatmap using Seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    cbar=False, annot_kws={"size": 14})
        
        ax.set_title(f"{model_name}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Label\n(0: Normal, 1: Fraud)', fontsize=12)
        ax.set_ylabel('True Label\n(0: Normal, 1: Fraud)', fontsize=12)
        
    plt.tight_layout()
    save_path = 'paper_confusion_matrices.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Success! High-resolution confusion matrices saved as '{save_path}'")


# Run the functions
if __name__ == "__main__":
    print("Generating High-Resolution Images for ...")
    create_comparison_chart()
    generate_confusion_matrices()
    
    print("\n All visualizations complete! ")
    plt.show() # Display all generated plots simultaneously at the end