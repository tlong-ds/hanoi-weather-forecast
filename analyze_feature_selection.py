"""
Feature Selection Analysis Script
==================================
Analyzes the results of the combined feature selection approach.
Provides insights into which features are important for short-term vs long-term forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
FEATURE_SELECTION_DIR = 'processed_data/feature_selection'
PLOTS_DIR = 'plots/feature_selection'

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_feature_data():
    """Load all feature selection data."""
    data = {}
    
    # Load feature lists
    data['selected'] = pd.read_csv(os.path.join(FEATURE_SELECTION_DIR, 'selected_features.csv'))
    data['short_term'] = pd.read_csv(os.path.join(FEATURE_SELECTION_DIR, 'short_term_features.csv'))
    data['long_term'] = pd.read_csv(os.path.join(FEATURE_SELECTION_DIR, 'long_term_features.csv'))
    
    # Load scores
    data['short_scores'] = pd.read_csv(os.path.join(FEATURE_SELECTION_DIR, 'short_term_scores.csv'), index_col=0)
    data['long_scores'] = pd.read_csv(os.path.join(FEATURE_SELECTION_DIR, 'long_term_scores.csv'), index_col=0)
    
    return data

def analyze_feature_overlap(data):
    """Analyze the overlap between short-term and long-term features."""
    short_features = set(data['short_term']['feature'])
    long_features = set(data['long_term']['feature'])
    
    shared = short_features & long_features
    short_only = short_features - long_features
    long_only = long_features - short_features
    
    print(f"\n{'='*70}")
    print("FEATURE OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"Total short-term features:  {len(short_features)}")
    print(f"Total long-term features:   {len(long_features)}")
    print(f"Shared features:            {len(shared)} ({len(shared)/len(short_features)*100:.1f}%)")
    print(f"Short-term only:            {len(short_only)}")
    print(f"Long-term only:             {len(long_only)}")
    print(f"{'='*70}\n")
    
    # Create Venn diagram data
    venn_data = {
        'Category': ['Short-term only', 'Shared', 'Long-term only'],
        'Count': [len(short_only), len(shared), len(long_only)]
    }
    
    # Plot Venn-style bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    bars = ax.bar(venn_data['Category'], venn_data['Count'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Features', fontsize=12)
    ax.set_title('Feature Selection: Short-term vs Long-term Overlap', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_overlap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOTS_DIR}/feature_overlap.png")
    
    return shared, short_only, long_only

def plot_top_features_comparison(data, top_n=20):
    """Compare top features for short-term vs long-term."""
    short_scores = data['short_scores'].head(top_n)
    long_scores = data['long_scores'].head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Short-term features
    ax1 = axes[0]
    y_pos = np.arange(len(short_scores))
    ax1.barh(y_pos, short_scores['rf_importance'], color='#ff9999', alpha=0.7, label='RF Importance')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_scores.index, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Random Forest Importance', fontsize=11)
    ax1.set_title(f'Top {top_n} Features for Short-Term (t+1)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Long-term features
    ax2 = axes[1]
    y_pos = np.arange(len(long_scores))
    ax2.barh(y_pos, long_scores['rf_importance'], color='#99ff99', alpha=0.7, label='RF Importance')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(long_scores.index, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Random Forest Importance', fontsize=11)
    ax2.set_title(f'Top {top_n} Features for Long-Term (t+N)', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_features_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOTS_DIR}/top_features_comparison.png")

def plot_feature_importance_methods(data, target='short'):
    """Compare different feature importance methods."""
    if target == 'short':
        scores = data['short_scores'].head(15)
        title = 'Short-Term (t+1) Feature Importance'
        filename = 'short_term_methods.png'
        color = '#ff9999'
    else:
        scores = data['long_scores'].head(15)
        title = 'Long-Term (t+N) Feature Importance'
        filename = 'long_term_methods.png'
        color = '#99ff99'
    
    # Normalize scores to 0-1 for comparison
    scores_norm = scores[['correlation', 'mutual_info', 'lasso_coef', 'rf_importance']].copy()
    for col in scores_norm.columns:
        scores_norm[col] = (scores_norm[col] - scores_norm[col].min()) / (scores_norm[col].max() - scores_norm[col].min() + 1e-10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(scores_norm))
    width = 0.2
    
    ax.barh(x - 1.5*width, scores_norm['correlation'], width, label='Correlation', alpha=0.8)
    ax.barh(x - 0.5*width, scores_norm['mutual_info'], width, label='Mutual Info', alpha=0.8)
    ax.barh(x + 0.5*width, scores_norm['lasso_coef'], width, label='Lasso Coef', alpha=0.8)
    ax.barh(x + 1.5*width, scores_norm['rf_importance'], width, label='RF Importance', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(scores_norm.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOTS_DIR}/{filename}")

def analyze_feature_types(data):
    """Analyze feature types (lag, rolling, temporal, etc.)."""
    selected_features = data['selected']['feature']
    
    # Categorize features
    categories = {
        'Lag Features': [],
        'Rolling Stats': [],
        'Temporal Features': [],
        'Cyclical Features': [],
        'Interaction Features': [],
        'Raw Features': []
    }
    
    for feat in selected_features:
        if '_lag' in feat:
            categories['Lag Features'].append(feat)
        elif '_roll_' in feat:
            categories['Rolling Stats'].append(feat)
        elif any(x in feat for x in ['year', 'is_weekend', 'is_summer', 'is_autumn', 'is_winter', 'is_spring']):
            categories['Temporal Features'].append(feat)
        elif any(x in feat for x in ['_sin', '_cos']):
            categories['Cyclical Features'].append(feat)
        elif any(x in feat for x in ['_sq', 'pressure_humidity', 'daylength_uv']):
            categories['Interaction Features'].append(feat)
        else:
            categories['Raw Features'].append(feat)
    
    # Print summary
    print(f"\n{'='*70}")
    print("FEATURE TYPE ANALYSIS")
    print(f"{'='*70}")
    for cat, feats in categories.items():
        if feats:
            print(f"{cat:.<25} {len(feats):>3} features")
            # Show examples
            for f in feats[:3]:
                print(f"  • {f}")
            if len(feats) > 3:
                print(f"  ... and {len(feats)-3} more")
    print(f"{'='*70}\n")
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_names = [k for k, v in categories.items() if v]
    cat_counts = [len(v) for k, v in categories.items() if v]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cat_names)))
    bars = ax.barh(cat_names, cat_counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_title('Selected Features by Category', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_types.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOTS_DIR}/feature_types.png")

def create_heatmap_comparison(data, top_n=25):
    """Create heatmap comparing feature importance scores."""
    # Get top features from both
    short_top = data['short_scores'].head(top_n)
    long_top = data['long_scores'].head(top_n)
    
    # Combine all unique features
    all_features = list(set(short_top.index.tolist() + long_top.index.tolist()))
    
    # Create comparison dataframe
    comparison = pd.DataFrame(index=all_features)
    comparison['Short-term Rank'] = short_top['mean_rank']
    comparison['Long-term Rank'] = long_top['mean_rank']
    comparison = comparison.fillna(999)  # Fill missing with high rank
    
    # Sort by average rank
    comparison['Avg Rank'] = comparison.mean(axis=1)
    comparison = comparison.sort_values('Avg Rank').head(top_n)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # Normalize for better visualization (lower rank = better)
    data_to_plot = comparison[['Short-term Rank', 'Long-term Rank']].copy()
    
    sns.heatmap(data_to_plot, annot=False, cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (lower is better)'},
                linewidths=0.5, ax=ax)
    
    ax.set_title(f'Feature Importance Ranks: Short-term vs Long-term (Top {top_n})', 
                 fontsize=13, fontweight='bold', pad=20)
    ax.set_ylabel('Features', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'rank_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {PLOTS_DIR}/rank_heatmap.png")

def main():
    """Main analysis function."""
    print(f"\n{'='*70}")
    print("FEATURE SELECTION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Ensure directories exist
    ensure_directories()
    
    # Check if feature selection data exists
    if not os.path.exists(FEATURE_SELECTION_DIR):
        print(f"❌ Feature selection directory not found: {FEATURE_SELECTION_DIR}")
        print("   Please run preprocessing.py first to generate feature selection data.")
        return
    
    # Load data
    print("Loading feature selection data...")
    try:
        data = load_feature_data()
        print("✓ Data loaded successfully\n")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("   Please run preprocessing.py first to generate feature selection data.")
        return
    
    # Run analyses
    print("Analyzing feature overlap...")
    shared, short_only, long_only = analyze_feature_overlap(data)
    
    print("\nAnalyzing feature types...")
    analyze_feature_types(data)
    
    print("\nGenerating comparison plots...")
    plot_top_features_comparison(data, top_n=20)
    
    print("\nGenerating method comparison plots...")
    plot_feature_importance_methods(data, target='short')
    plot_feature_importance_methods(data, target='long')
    
    print("\nGenerating rank heatmap...")
    create_heatmap_comparison(data, top_n=25)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"All plots saved to: {PLOTS_DIR}/")
    print("\nGenerated files:")
    print("  • feature_overlap.png - Venn-style overlap visualization")
    print("  • top_features_comparison.png - Side-by-side top features")
    print("  • short_term_methods.png - Method comparison for short-term")
    print("  • long_term_methods.png - Method comparison for long-term")
    print("  • feature_types.png - Feature category distribution")
    print("  • rank_heatmap.png - Feature rank comparison heatmap")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
