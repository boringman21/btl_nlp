# Water Leakage Visualization

This document provides information about the visualization capabilities added to the water leakage analysis project.

## Overview

The visualization module creates comprehensive dashboards and charts to help interpret the water leakage analysis results. It presents the outcomes from all four priority applications in an integrated visual format:

1. Leak Detection
2. Water Consumption Pattern Prediction
3. Early Warning System
4. Anomaly Detection

## Usage Options

There are two main ways to use the visualization functionality:

### 1. Visualize during analysis

Run the analysis with visualization enabled:

```bash
python analyze_priority_app.py --visualization --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2
```

Additional options:
- `--sensor_id SENSOR_ID`: Focus on a specific sensor
- `--save_results`: Save analysis results to file for later visualization
- `--output_dir OUTPUT_DIR`: Specify where to save visualizations (default: ./analysis_results)
- `--report_dir REPORT_DIR`: Specify where to save the report (default: ./report)

Example with all options:

```bash
python analyze_priority_app.py --visualization --save_results --sensor_id 8401210607558 --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2 --output_dir ./my_analysis --report_dir ./my_report
```

### 2. Visualize from saved results

If you've previously run the analysis with `--save_results`, you can create visualizations later using:

```bash
python visualize_water_leakage.py --results_file ./analysis_results/results/analysis_results.json
```

Additional options:
- `--sensor_id SENSOR_ID`: Focus on a specific sensor
- `--output_dir OUTPUT_DIR`: Specify where to save visualizations (default: ./visualization_output)

Example with all options:

```bash
python visualize_water_leakage.py --results_file ./analysis_results/results/analysis_results.json --sensor_id 8401210607558 --output_dir ./my_visualizations
```

## Visualization Outputs

The visualization module creates the following outputs:

1. **Sensor-specific dashboards**: A comprehensive dashboard for each selected sensor showing:
   - Leak detection summary
   - Water consumption patterns
   - Early warning indicators
   - Anomaly detection results
   - Summary metrics and KPIs

2. **Cross-sensor comparison**: Charts comparing different sensors:
   - Top sensors by leak count
   - Top sensors by anomaly percentage

All visualizations are saved in the specified output directory under a `dashboard` subfolder.

## Customizing Visualizations

The visualization module provides a clean, informative design by default. If you need to customize the visualizations:

1. Modify the `water_leakage/apps/visualize.py` file to change plot styles, colors, or layout
2. The module uses Matplotlib and Seaborn for the visualizations
3. Main plot functions are prefixed with `_plot_` and can be modified individually

## Troubleshooting

If you encounter issues:

1. **Visualization not appearing**: Check the output directory specified
2. **Missing data in visualizations**: Ensure the analysis completed successfully
3. **Error loading results file**: Make sure the path to the results file is correct

For further assistance, check the console output for specific error messages. 