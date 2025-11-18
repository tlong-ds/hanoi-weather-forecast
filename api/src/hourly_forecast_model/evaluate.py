import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import configuration
from src.hourly_forecast_model.helper import (
    PROJECT_ROOT, N_STEPS_AHEAD, TARGET_COLUMNS, MODELS_DIR, PLOTS_DIR,
    load_data
)

# Import PerHorizonWrapper to enable unpickling models trained with per-horizon approach
from src.hourly_forecast_model.train import PerHorizonWrapper

# Evaluation results directory
EVALUATE_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'src', 'hourly_forecast_model', 'evaluate_results')

# Evaluation settings
VERBOSE_EVALUATION = True
SAVE_PLOTS = True
PLOT_FORMAT = 'png'
PLOT_DPI = 300


def load_trained_model():
    """Load trained multi-output model."""
    model_path = os.path.join(MODELS_DIR, 'model_multioutput_24h.joblib')
    
    try:
        model = joblib.load(model_path)
        if VERBOSE_EVALUATION:
            print(f"✓ Model loaded from: {model_path}")
        return model
    except FileNotFoundError:
        print(f"✗ Model not found: {model_path}")
        print("  Run train.py first to train the model.")
        return None


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
    
    Returns:
        dict: Dictionary of metrics (MAE, RMSE, MAPE, R²)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_multioutput_model():
    """
    Evaluate multi-output model on test set.
    
    Returns:
        dict: Evaluation results with metrics per hour
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING MULTI-OUTPUT MODEL")
    print(f"{'='*70}")
    
    # Load model
    model = load_trained_model()
    if model is None:
        return None
    
    # Load test data
    print("\nLoading test data...")
    _, _, _, _, X_test, y_test = load_data()
    
    if X_test is None:
        print("✗ Failed to load test data")
        return None
    
    print(f"✓ Test data loaded:")
    print(f"  Samples: {X_test.shape[0]}")
    print(f"  Features: {X_test.shape[1]}")
    print(f"  Targets: {y_test.shape[1]} hours")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    print(f"✓ Predictions complete: {y_pred.shape}")
    
    # Calculate metrics per hour
    print(f"\n{'='*70}")
    print("PER-HOUR METRICS")
    print(f"{'='*70}")
    
    results_per_hour = {}
    
    for i in range(y_test.shape[1]):
        hour = i + 1
        hour_name = f"t+{hour}h"
        
        y_true_hour = y_test.iloc[:, i]
        y_pred_hour = y_pred[:, i]
        
        metrics = calculate_metrics(y_true_hour, y_pred_hour)
        
        results_per_hour[hour_name] = {
            'hour': hour,
            'metrics': metrics,
            'predictions': y_pred_hour,
            'actuals': y_true_hour.values
        }
        
        if VERBOSE_EVALUATION:
            print(f"\n{hour_name}:")
            print(f"  MAE:  {metrics['MAE']:.4f}°C")
            print(f"  RMSE: {metrics['RMSE']:.4f}°C")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  R²:   {metrics['R2']:.4f}")
    
    # Calculate average metrics
    avg_mae = np.mean([r['metrics']['MAE'] for r in results_per_hour.values()])
    avg_rmse = np.mean([r['metrics']['RMSE'] for r in results_per_hour.values()])
    avg_mape = np.mean([r['metrics']['MAPE'] for r in results_per_hour.values()])
    avg_r2 = np.mean([r['metrics']['R2'] for r in results_per_hour.values()])
    
    print(f"\n{'='*70}")
    print("OVERALL METRICS (Average across all hours)")
    print(f"{'='*70}")
    print(f"Average MAE:  {avg_mae:.4f}°C")
    print(f"Average RMSE: {avg_rmse:.4f}°C")
    print(f"Average MAPE: {avg_mape:.2f}%")
    print(f"Average R²:   {avg_r2:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'per_hour': results_per_hour,
        'average': {
            'MAE': avg_mae,
            'RMSE': avg_rmse,
            'MAPE': avg_mape,
            'R2': avg_r2
        }
    }


def plot_predictions_vs_actuals(results, hour_name, save_path):
    """Plot predicted vs actual values for a specific hour."""
    hour_results = results['per_hour'][hour_name]
    y_test = hour_results['actuals']
    y_pred = hour_results['predictions']
    metrics = hour_results['metrics']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title(f'Predictions vs Actuals - {hour_name}\n'
              f'RMSE: {metrics["RMSE"]:.4f}°C, R²: {metrics["R2"]:.4f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT)
        if VERBOSE_EVALUATION:
            print(f"  ✓ Plot saved: {save_path}")
    
    plt.close()


def plot_metrics_comparison(results, save_dir):
    """Plot comparison of metrics across all hours."""
    hours = [r['hour'] for r in results['per_hour'].values()]
    mae_values = [r['metrics']['MAE'] for r in results['per_hour'].values()]
    rmse_values = [r['metrics']['RMSE'] for r in results['per_hour'].values()]
    r2_values = [r['metrics']['R2'] for r in results['per_hour'].values()]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE plot
    axes[0].bar(hours, mae_values, color='steelblue', alpha=0.7)
    axes[0].axhline(y=results['average']['MAE'], color='red', linestyle='--', 
                    label=f'Average: {results["average"]["MAE"]:.4f}°C')
    axes[0].set_xlabel('Hour Ahead', fontsize=12)
    axes[0].set_ylabel('MAE (°C)', fontsize=12)
    axes[0].set_title('Mean Absolute Error by Hour', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE plot
    axes[1].bar(hours, rmse_values, color='coral', alpha=0.7)
    axes[1].axhline(y=results['average']['RMSE'], color='red', linestyle='--',
                    label=f'Average: {results["average"]["RMSE"]:.4f}°C')
    axes[1].set_xlabel('Hour Ahead', fontsize=12)
    axes[1].set_ylabel('RMSE (°C)', fontsize=12)
    axes[1].set_title('Root Mean Squared Error by Hour', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # R² plot
    axes[2].bar(hours, r2_values, color='seagreen', alpha=0.7)
    axes[2].axhline(y=results['average']['R2'], color='red', linestyle='--',
                    label=f'Average: {results["average"]["R2"]:.4f}')
    axes[2].set_xlabel('Hour Ahead', fontsize=12)
    axes[2].set_ylabel('R²', fontsize=12)
    axes[2].set_title('R² Score by Hour', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'metrics_comparison.{PLOT_FORMAT}')
    if SAVE_PLOTS:
        plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT)
        if VERBOSE_EVALUATION:
            print(f"  ✓ Metrics comparison plot saved: {save_path}")
    
    plt.close()


def plot_time_series_predictions(results, save_dir, n_samples=500):
    """
    Plot time series comparison of predictions vs actuals for multiple horizons.
    Shows temperature fluctuations over time to visualize model performance.
    
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save plots
        n_samples: Number of samples to plot (default 500 for clarity)
    """
    # Select representative horizons to plot
    horizons_to_plot = ['t+1h', 't+6h', 't+12h', 't+18h', 't+24h']
    
    # Create multi-panel plot
    fig, axes = plt.subplots(len(horizons_to_plot), 1, figsize=(16, 12))
    
    for idx, hour_name in enumerate(horizons_to_plot):
        hour_results = results['per_hour'][hour_name]
        y_test = hour_results['actuals'][:n_samples]
        y_pred = hour_results['predictions'][:n_samples]
        metrics = hour_results['metrics']
        
        ax = axes[idx]
        
        # Plot actual and predicted values
        x = np.arange(len(y_test))
        ax.plot(x, y_test, label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.8)
        ax.plot(x, y_pred, label='Predicted', color='#A23B72', linewidth=1.5, alpha=0.8, linestyle='--')
        
        # Fill area between predictions and actuals to show errors
        ax.fill_between(x, y_test, y_pred, alpha=0.2, color='gray')
        
        # Styling
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'{hour_name} - RMSE: {metrics["RMSE"]:.4f}°C, R²: {metrics["R2"]:.4f}', 
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Only show x-label on bottom plot
        if idx == len(horizons_to_plot) - 1:
            ax.set_xlabel('Sample Index', fontsize=10)
    
    plt.suptitle('Hourly Temperature Forecast: Time Series Comparison\n(Predictions vs Actuals)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = os.path.join(save_dir, f'time_series_predictions.{PLOT_FORMAT}')
    if SAVE_PLOTS:
        plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
        if VERBOSE_EVALUATION:
            print(f"  ✓ Time series plot saved: {save_path}")
    
    plt.close()
    
    # Create detailed single-horizon plot for t+1h (highest frequency fluctuations)
    fig, ax = plt.subplots(figsize=(18, 6))
    
    hour_name = 't+1h'
    hour_results = results['per_hour'][hour_name]
    y_test = hour_results['actuals'][:n_samples]
    y_pred = hour_results['predictions'][:n_samples]
    metrics = hour_results['metrics']
    
    x = np.arange(len(y_test))
    ax.plot(x, y_test, label='Actual Temperature', color='#2E86AB', linewidth=2, marker='o', 
            markersize=3, alpha=0.7)
    ax.plot(x, y_pred, label='Predicted Temperature', color='#A23B72', linewidth=2, marker='s', 
            markersize=3, alpha=0.7, linestyle='--')
    
    # Calculate and plot error bars
    errors = np.abs(y_test - y_pred)
    ax.fill_between(x, y_test - errors, y_test + errors, alpha=0.15, color='red', 
                     label='Prediction Error Range')
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Detailed Time Series: 1-Hour Ahead Forecast\n'
                 f'RMSE: {metrics["RMSE"]:.4f}°C | MAE: {metrics["MAE"]:.4f}°C | R²: {metrics["R2"]:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.7)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'time_series_detailed_t1h.{PLOT_FORMAT}')
    if SAVE_PLOTS:
        plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
        if VERBOSE_EVALUATION:
            print(f"  ✓ Detailed time series plot saved: {save_path}")
    
    plt.close()


def save_evaluation_results(results):
    """Save evaluation results to JSON file."""
    # Convert results to serializable format
    results_dict = {
        'average_metrics': results['average'],
        'per_hour_metrics': {}
    }
    
    for hour_name, result in results['per_hour'].items():
        results_dict['per_hour_metrics'][hour_name] = {
            'hour': result['hour'],
            'metrics': result['metrics']
        }
    
    # Save to JSON
    results_path = os.path.join(EVALUATE_RESULTS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✓ Evaluation results saved to: {results_path}")


def main():
    """Main evaluation function."""
    print("\n" + "="*70)
    print("HOURLY FORECAST MULTI-OUTPUT MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Create results directory
    os.makedirs(EVALUATE_RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Evaluate model
    results = evaluate_multioutput_model()
    
    if results is None:
        print("❌ Evaluation failed.")
        return
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    # Save scatter plots for select hours
    for hour in [1, 6, 12, 18, 24]:
        hour_name = f"t+{hour}h"
        plot_path = os.path.join(PLOTS_DIR, f'predictions_vs_actuals_{hour_name}.{PLOT_FORMAT}')
        plot_predictions_vs_actuals(results, hour_name, plot_path)
    
    # Save metrics comparison plot
    plot_metrics_comparison(results, PLOTS_DIR)
    
    # Save time series plots showing temperature fluctuations
    plot_time_series_predictions(results, PLOTS_DIR)
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    save_evaluation_results(results)
    
    print("\n✅ Evaluation complete!")
    print(f"   Results saved to: {EVALUATE_RESULTS_DIR}")
    print(f"   Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
