import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import configuration
from src.daily_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, MODELS_DIR, PLOTS_DIR,
    SAVE_PLOTS, PLOT_FORMAT, PLOT_DPI, VERBOSE_EVALUATION
)

# Evaluation results directory
EVALUATE_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'daily_forecast_model', 'evaluate_results')


def load_test_data_for_target(day_step):
    """
    Load test data for specific target day.
    
    Args:
        day_step (int): Target day (1-5)
    
    Returns:
        tuple: (X_test, y_test) or (None, None) on error
    """
    day_str = f"t_{day_step}"
    data_dir = os.path.join(PROJECT_ROOT, 'processed_data', f'target_{day_str}')
    
    try:
        X_test = pd.read_csv(os.path.join(data_dir, f'X_test_t{day_step}.csv'), index_col=0)
        y_test = pd.read_csv(os.path.join(data_dir, f'y_test_t{day_step}.csv'), index_col=0)
        
        # Convert y to 1D array
        y_test = y_test.values.ravel()
        
        if VERBOSE_EVALUATION:
            print(f"  ✓ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        return X_test, y_test
    
    except FileNotFoundError as e:
        print(f"  ✗ Test data not found in '{data_dir}'")
        print(f"    Error: {e}")
        return None, None


def load_trained_model(target_name):
    """
    Load trained model for specific target.
    
    Args:
        target_name (str): Target name (e.g., 't+1', 't+2')
    
    Returns:
        model: Loaded model or None on error
    """
    model_path = os.path.join(MODELS_DIR, f"model_{target_name}.joblib")
    
    try:
        model = joblib.load(model_path)
        if VERBOSE_EVALUATION:
            print(f"  ✓ Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"  ✗ Model not found: {model_path}")
        return None


def calculate_metrics(y_true, y_pred, target_name):
    """
    Calculate evaluation metrics for a single target.
    
    Args:
        y_true (np.array): True values
        y_pred (np.array): Predicted values
        target_name (str): Target name for display
    
    Returns:
        dict: Dictionary of metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }
    
    print(f"\n{'-'*70}")
    print(f"Metrics for {target_name}:")
    print(f"  MAE:  {mae:8.4f}°C")
    print(f"  RMSE: {rmse:8.4f}°C")
    print(f"  MAPE: {mape:8.2f}%")
    print(f"  R²:   {r2:8.4f}")
    print(f"{'-'*70}")
    
    return metrics


def evaluate_per_target_models():
    """
    Evaluate all trained models on test set.
    
    Returns:
        dict: Dictionary of results {target_name: {metrics, predictions}}
    """
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"EVALUATING PER-TARGET MODELS ON TEST SET")
    print(f"{'='*70}\n")
    
    for day_step in range(1, N_STEPS_AHEAD + 1):
        target_name = f"t+{day_step}"
        
        print(f"{'='*70}")
        print(f"[{target_name}]")
        print(f"{'='*70}")
        
        # Load model
        model = load_trained_model(target_name)
        if model is None:
            print(f"  ✗ Skipping {target_name} - model not found")
            continue
        
        # Load test data
        X_test, y_test = load_test_data_for_target(day_step)
        if X_test is None:
            print(f"  ✗ Skipping {target_name} - test data not found")
            continue
        
        # Make predictions
        print(f"  Predicting...", end="", flush=True)
        y_pred = model.predict(X_test)
        print(f" ✓ Complete")
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, target_name)
        
        # Store results
        all_results[target_name] = {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'n_samples': len(y_test)
        }
        
        print()
    
    return all_results


def print_summary(all_results):
    """Print evaluation summary across all targets."""
    if not all_results:
        print("No results to summarize.")
        return
    
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}\n")
    
    # Create summary table
    print(f"{'Target':<10} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
    print(f"{'-'*70}")
    
    total_mae = 0
    total_rmse = 0
    total_mape = 0
    total_r2 = 0
    
    for target_name, results in all_results.items():
        metrics = results['metrics']
        print(f"{target_name:<10} {metrics['MAE']:<10.4f} {metrics['RMSE']:<10.4f} "
              f"{metrics['MAPE']:<10.2f} {metrics['R2']:<10.4f}")
        
        total_mae += metrics['MAE']
        total_rmse += metrics['RMSE']
        total_mape += metrics['MAPE']
        total_r2 += metrics['R2']
    
    n_targets = len(all_results)
    avg_mae = total_mae / n_targets
    avg_rmse = total_rmse / n_targets
    avg_mape = total_mape / n_targets
    avg_r2 = total_r2 / n_targets
    
    print(f"{'-'*70}")
    print(f"{'Average':<10} {avg_mae:<10.4f} {avg_rmse:<10.4f} "
          f"{avg_mape:<10.2f} {avg_r2:<10.4f}")
    print(f"{'='*70}\n")


def save_results(all_results):
    """Save evaluation results to JSON and CSV files."""
    if not all_results:
        return
    
    # Create evaluate_results directory if it doesn't exist
    os.makedirs(EVALUATE_RESULTS_DIR, exist_ok=True)
    
    # Prepare results for JSON (convert numpy arrays to lists)
    json_results = {}
    for target_name, results in all_results.items():
        json_results[target_name] = {
            'metrics': results['metrics'],
            'n_samples': results['n_samples']
        }
    
    # Save to JSON
    results_json_path = os.path.join(EVALUATE_RESULTS_DIR, 'evaluation_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Results saved to: {results_json_path}")
    
    # Save metrics to CSV
    metrics_data = []
    for target_name, results in all_results.items():
        row = {'target': target_name}
        row.update(results['metrics'])
        row['n_samples'] = results['n_samples']
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(EVALUATE_RESULTS_DIR, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"✓ Metrics saved to: {metrics_csv_path}")


def create_visualizations(all_results):
    """Create scatter plots and time series comparisons."""
    if not all_results:
        return
    
    # Create plots directory inside evaluate_results
    plots_dir = os.path.join(EVALUATE_RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Scatter plot for each target
    print("Creating scatter plots...")
    for target_name, results in all_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        metrics = results['metrics']
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, s=30)
        
        # Perfect prediction line
        min_val = y_true.min()
        max_val = y_true.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.title(f'{target_name} - Actual vs Predicted Temperature\n'
                  f'RMSE={metrics["RMSE"]:.4f}°C, R²={metrics["R2"]:.4f}',
                  fontsize=12, fontweight='bold')
        plt.xlabel('Actual Temperature (°C)', fontsize=11)
        plt.ylabel('Predicted Temperature (°C)', fontsize=11)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if SAVE_PLOTS:
            plot_path = os.path.join(plots_dir, f'scatter_{target_name}.{PLOT_FORMAT}')
            plt.savefig(plot_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
            print(f"  ✓ Scatter {target_name}: {plot_path}")
        
        plt.close()
    
    # Combined scatter plot
    plt.figure(figsize=(14, 10))
    for target_name, results in all_results.items():
        plt.scatter(results['y_true'], results['y_pred'], 
                   alpha=0.4, s=20, label=target_name)
    
    # Perfect prediction line
    all_true = np.concatenate([r['y_true'] for r in all_results.values()])
    min_val = all_true.min()
    max_val = all_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title('All Targets - Actual vs Predicted Temperature', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        combined_path = os.path.join(plots_dir, f'scatter_all_targets.{PLOT_FORMAT}')
        plt.savefig(combined_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
        print(f"  ✓ Scatter combined: {combined_path}")
    
    plt.close()
    
    # Time series plots for each target
    print("\nCreating time series plots...")
    create_timeseries_plots(all_results)
    
    print(f"{'='*70}\n")


def create_timeseries_plots(all_results):
    """Create time series comparison plots for actual vs predicted values."""
    
    # Create plots directory inside evaluate_results
    plots_dir = os.path.join(EVALUATE_RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Individual time series for each target
    for target_name, results in all_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        metrics = results['metrics']
        
        # Get indices for x-axis (sample numbers)
        indices = np.arange(len(y_true))
        
        # Plot full time series
        plt.figure(figsize=(16, 6))
        plt.plot(indices, y_true, 'b-', linewidth=1.5, alpha=0.7, label='Actual')
        plt.plot(indices, y_pred, 'r-', linewidth=1.5, alpha=0.7, label='Predicted')
        
        plt.title(f'{target_name} - Time Series Comparison\n'
                  f'RMSE={metrics["RMSE"]:.4f}°C, MAE={metrics["MAE"]:.4f}°C',
                  fontsize=13, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=11)
        plt.ylabel('Temperature (°C)', fontsize=11)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if SAVE_PLOTS:
            ts_path = os.path.join(plots_dir, f'timeseries_{target_name}.{PLOT_FORMAT}')
            plt.savefig(ts_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
            print(f"  ✓ Time series {target_name}: {ts_path}")
        
        plt.close()
        
        # Plot zoomed-in view (first 200 samples)
        n_samples = min(200, len(y_true))
        
        plt.figure(figsize=(16, 6))
        plt.plot(indices[:n_samples], y_true[:n_samples], 'b-', linewidth=2, alpha=0.7, label='Actual', marker='o', markersize=3)
        plt.plot(indices[:n_samples], y_pred[:n_samples], 'r-', linewidth=2, alpha=0.7, label='Predicted', marker='s', markersize=3)
        
        plt.title(f'{target_name} - Time Series Comparison (First {n_samples} Samples)\n'
                  f'RMSE={metrics["RMSE"]:.4f}°C, MAE={metrics["MAE"]:.4f}°C',
                  fontsize=13, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=11)
        plt.ylabel('Temperature (°C)', fontsize=11)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if SAVE_PLOTS:
            ts_zoom_path = os.path.join(plots_dir, f'timeseries_{target_name}_zoom.{PLOT_FORMAT}')
            plt.savefig(ts_zoom_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
            print(f"  ✓ Time series zoom {target_name}: {ts_zoom_path}")
        
        plt.close()
    
    # Combined time series plot (all targets on one chart)
    n_samples = min(200, len(list(all_results.values())[0]['y_true']))
    
    fig, axes = plt.subplots(N_STEPS_AHEAD, 1, figsize=(16, 3*N_STEPS_AHEAD))
    if N_STEPS_AHEAD == 1:
        axes = [axes]
    
    for idx, (target_name, results) in enumerate(all_results.items()):
        y_true = results['y_true'][:n_samples]
        y_pred = results['y_pred'][:n_samples]
        metrics = results['metrics']
        indices = np.arange(len(y_true))
        
        ax = axes[idx]
        ax.plot(indices, y_true, 'b-', linewidth=1.5, alpha=0.7, label='Actual', marker='o', markersize=2)
        ax.plot(indices, y_pred, 'r-', linewidth=1.5, alpha=0.7, label='Predicted', marker='s', markersize=2)
        
        ax.set_title(f'{target_name} - RMSE={metrics["RMSE"]:.4f}°C, MAE={metrics["MAE"]:.4f}°C',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Temp (°C)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'All Targets - Time Series Comparison (First {n_samples} Samples)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        combined_ts_path = os.path.join(plots_dir, f'timeseries_all_targets.{PLOT_FORMAT}')
        plt.savefig(combined_ts_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
        print(f"  ✓ Time series combined: {combined_ts_path}")
    
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PER-TARGET MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Evaluate all models
    all_results = evaluate_per_target_models()
    
    if all_results:
        # Print summary
        print_summary(all_results)
        
        # Save results
        save_results(all_results)
        
        # Create visualizations
        create_visualizations(all_results)
        
        print("✅ Evaluation complete!")
    else:
        print("❌ No models were evaluated successfully.")
