import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import configuration and helper functions
from daily_forecast_model.helper import (
    TARGET_COLUMNS, SAVE_RESULTS, RESULTS_FILENAME, RESULTS_DIR,
    PLOT_ALL_HORIZONS, HORIZONS_TO_PLOT, VERBOSE_EVALUATION,
    PRINT_METRICS_PER_HORIZON, PLOT_FIGSIZE_SCATTER, PLOT_FIGSIZE_TIMESERIES,
    MODELS_DIR, get_enabled_models, SAVE_PLOTS, PLOTS_DIR, PLOT_FORMAT, PLOT_DPI,
    load_data, load_trained_models
)


def calculate_metrics(y_true, y_pred_array, model_name="Model", is_test_set=False):
    """
    Calculate evaluation metrics for multi-step forecasting.
    
    For DEV set: MAE, RMSE, MAPE
    For TEST set: MAE, RMSE, MAPE, R²
    
    Args:
        y_true (pd.DataFrame): DataFrame with actual values
        y_pred_array (np.array): Predicted values from model.predict()
        model_name (str): Model name for display
        is_test_set (bool): If True, compute R² metric

    Returns:
        tuple: (model_results dict, y_pred_df DataFrame)
    """
    # Convert predictions to DataFrame
    y_pred_df = pd.DataFrame(
        y_pred_array, 
        index=y_true.index, 
        columns=TARGET_COLUMNS
    )
    
    model_results = {}
    data_type = "Test" if is_test_set else "Dev"
    
    print(f"\n{'='*70}")
    print(f"Evaluation Results ({data_type} Set) - {model_name}")
    print(f"{'='*70}")
    
    # Calculate metrics for each forecast horizon
    if PRINT_METRICS_PER_HORIZON:
        print(f"\nPer-Horizon Metrics:")
        print(f"{'-'*70}")
    
    for horizon in TARGET_COLUMNS:
        mae = mean_absolute_error(y_true[horizon], y_pred_df[horizon])
        mse = mean_squared_error(y_true[horizon], y_pred_df[horizon])
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true[horizon] - y_pred_df[horizon]) / y_true[horizon])) * 100
        
        model_results[f'{horizon}_MAE'] = mae
        model_results[f'{horizon}_RMSE'] = rmse
        model_results[f'{horizon}_MAPE'] = mape
        
        if PRINT_METRICS_PER_HORIZON:
            print(f"  {horizon}:")
            print(f"    MAE:  {mae:8.4f}°C")
            print(f"    RMSE: {rmse:8.4f}°C")
            print(f"    MAPE: {mape:8.2f}%")
        
        # Add R² only for test set
        if is_test_set:
            r2 = r2_score(y_true[horizon], y_pred_df[horizon])
            model_results[f'{horizon}_R2'] = r2
            if PRINT_METRICS_PER_HORIZON:
                print(f"    R²:   {r2:8.4f}")

    # Calculate average metrics across all horizons
    avg_mae = mean_absolute_error(y_true.values.flatten(), y_pred_df.values.flatten())
    avg_mse = mean_squared_error(y_true.values.flatten(), y_pred_df.values.flatten())
    avg_rmse = np.sqrt(avg_mse)
    avg_mape = np.mean(np.abs((y_true.values.flatten() - y_pred_df.values.flatten()) / y_true.values.flatten())) * 100
    
    model_results["Average_MAE"] = avg_mae
    model_results["Average_RMSE"] = avg_rmse
    model_results["Average_MAPE"] = avg_mape
    
    print(f"\n{'-'*70}")
    print(f"Average Across All Horizons:")
    print(f"  MAE:  {avg_mae:8.4f}°C")
    print(f"  RMSE: {avg_rmse:8.4f}°C")
    print(f"  MAPE: {avg_mape:8.2f}%")
    
    # Add average R² only for test set
    if is_test_set:
        avg_r2 = r2_score(y_true.values.flatten(), y_pred_df.values.flatten())
        model_results["Average_R2"] = avg_r2
        print(f"  R²:   {avg_r2:8.4f}")
    
    print(f"{'-'*70}\n")

    return model_results, y_pred_df

def visualize_results(y_true, all_predictions):
    """
    Create scatter and line plots comparing actual vs predicted values.
    Can show all horizons or selected horizons based on config.
    Optionally saves plots to disk.

    Args:
        y_true (pd.DataFrame): DataFrame with actual values
        all_predictions (dict): Dictionary {'model_name': y_pred_df}
    """
    plt.ioff()  # Turn off interactive mode for ClearML integration
    
    if VERBOSE_EVALUATION:
        print(f"\n✓ Creating visualizations...")
    
    # Determine which horizons to plot
    if HORIZONS_TO_PLOT is not None:
        horizons_to_plot = HORIZONS_TO_PLOT
    elif PLOT_ALL_HORIZONS:
        horizons_to_plot = TARGET_COLUMNS
    else:
        # Default: first and last horizon only
        horizons_to_plot = [TARGET_COLUMNS[0], TARGET_COLUMNS[-1]]
    
    for idx, horizon in enumerate(horizons_to_plot, 1):
        # ===== Scatter Plot =====
        fig_scatter = plt.figure(figsize=PLOT_FIGSIZE_SCATTER)
        
        for model_name, y_pred_df in all_predictions.items():
            plt.scatter(y_true[horizon], y_pred_df[horizon], alpha=0.5, label=model_name, s=20)
        
        # Perfect prediction reference line
        min_val = y_true[horizon].min()
        max_val = y_true[horizon].max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.title(f'Actual vs Predicted Temperature - {horizon}', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Temperature (°C)', fontsize=12)
        plt.ylabel('Predicted Temperature (°C)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save scatter plot
        if SAVE_PLOTS:
            scatter_filename = f"scatter_{horizon}.{PLOT_FORMAT}"
            scatter_path = os.path.join(PLOTS_DIR, scatter_filename)
            plt.savefig(scatter_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
            if VERBOSE_EVALUATION:
                print(f"  ✓ Scatter plot saved: {scatter_path}")
        else:
            if VERBOSE_EVALUATION:
                print(f"  • Scatter plot: {horizon}")
        
        plt.close(fig_scatter)
        
        # ===== Time Series Line Plot =====
        fig_ts = plt.figure(figsize=PLOT_FIGSIZE_TIMESERIES)
        
        # Plot actual values
        plt.plot(y_true.index, y_true[horizon], 
                label=f'Actual ({horizon})', 
                color='black', linewidth=1, marker='o', markersize=3)
        
        # Plot predictions from each model
        for model_name, y_pred_df in all_predictions.items():
            plt.plot(y_pred_df.index, y_pred_df[horizon], 
                    label=f'Predicted ({model_name})', 
                    linestyle='--', linewidth=1.5, alpha=0.8, marker='.')
        
        plt.title(f'Temperature Forecasting Over Time - {horizon}', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save time series plot
        if SAVE_PLOTS:
            ts_filename = f"timeseries_{horizon}.{PLOT_FORMAT}"
            ts_path = os.path.join(PLOTS_DIR, ts_filename)
            plt.savefig(ts_path, dpi=PLOT_DPI, format=PLOT_FORMAT, bbox_inches='tight')
            if VERBOSE_EVALUATION:
                print(f"  ✓ Time series plot saved: {ts_path}")
        else:
            if VERBOSE_EVALUATION:
                print(f"  • Time series plot: {horizon}")
        
        plt.close(fig_ts)
    
    if SAVE_PLOTS and VERBOSE_EVALUATION:
        print(f"\n✓ All plots saved to '{PLOTS_DIR}/' directory")

def save_results_summary(results, output_filename=RESULTS_FILENAME, is_test_set=False):
    """
    Consolidate results from dictionary and save to CSV.

    Args:
        results (dict): Nested dictionary, e.g., {'Model 1': {'MAE': 1.0, 'RMSE': 2.0}}
        output_filename (str): Output CSV filename
        is_test_set (bool): If True, include R² columns
    """
    if not SAVE_RESULTS:
        return
    
    if VERBOSE_EVALUATION:
        print(f"\n✓ Summarizing Results:")
    
    result_df = pd.DataFrame(results).T
    
    # Order columns for readability
    cols_ordered = []
    for h in TARGET_COLUMNS:
        cols_ordered.append(f'{h}_MAE')
        cols_ordered.append(f'{h}_RMSE')
        cols_ordered.append(f'{h}_MAPE')
        if is_test_set:
            cols_ordered.append(f'{h}_R2')
    
    cols_ordered.extend(['Average_MAE', 'Average_RMSE', 'Average_MAPE'])
    if is_test_set:
        cols_ordered.append('Average_R2')
    
    final_cols = [c for c in cols_ordered if c in result_df.columns]
    
    print(result_df[final_cols].round(4).to_string())
    
    # Save to file
    output_path = os.path.join(RESULTS_DIR, output_filename)
    result_df[final_cols].round(4).to_csv(output_path)
    
    if VERBOSE_EVALUATION:
        print(f"\n✓ Results saved to '{output_path}'")

if __name__ == "__main__":
    # Example usage (assuming data loading functions are defined elsewhere)
    X_test, y_test = load_data('test')
    
    trained_models = load_trained_models()  # Placeholder function to load models
    all_model_predictions = {}
    all_model_results = {}
    
    # Example model evaluation loop
    for model_name, model in trained_models.items():
        y_test_pred = model.predict(X_test)
        test_results, y_test_pred_df = calculate_metrics(y_test, y_test_pred, model_name=model_name, is_test_set=True)
        all_model_results[f"{model_name}_Test"] = test_results
        all_model_predictions[f"{model_name}_Test"] = y_test_pred_df
    
    visualize_results(y_test, all_model_predictions)
    save_results_summary(all_model_results, output_filename=RESULTS_FILENAME, is_test_set=True)