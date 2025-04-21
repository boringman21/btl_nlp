#!/usr/bin/env python3
"""
Script để xử lý tự động tất cả các cảm biến có 2 điểm đo áp suất.
Tự động phát hiện, dự đoán và trực quan hóa kết quả cho các cảm biến.
"""

import os
import subprocess
import sys
import pandas as pd
import json
from datetime import datetime

from water_leakage.data.data_loader import load_data
from water_leakage.data.data_transform import transform_df


def get_sensor_ids_with_both_pressures(df):
    """Lấy danh sách các cảm biến có cả 2 điểm đo áp suất."""
    filtered_columns = df.columns[df.columns.str.endswith('_Pressure_2')]
    return [col.split('_')[0] for col in filtered_columns if f"{col.split('_')[0]}_Flow" in df.columns]


def main():
    """Chức năng chính để xử lý các cảm biến có 2 áp suất."""
    # Đường dẫn mặc định
    data_path = "./LeakDataset/Logger_Data_2024_Bau_Bang-2"
    output_dir = "./prediction_results_both_pressures"
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Đang tải dữ liệu từ: {data_path}")
    
    # Tải và chuyển đổi dữ liệu
    merged_df = load_data(data_path)
    if merged_df is None or merged_df.empty:
        print("Không thể tải dữ liệu.")
        return 1
    
    result_df = transform_df(merged_df)
    
    # Lấy danh sách cảm biến có 2 áp suất
    sensors_with_both_pressures = get_sensor_ids_with_both_pressures(result_df)
    
    print(f"Tìm thấy {len(sensors_with_both_pressures)} cảm biến có 2 điểm đo áp suất:")
    for i, sensor_id in enumerate(sensors_with_both_pressures):
        print(f"{i+1}. {sensor_id}")
    
    # Ghi các cảm biến vào file
    sensors_file = os.path.join(output_dir, "sensors_with_both_pressures.json")
    with open(sensors_file, 'w', encoding='utf-8') as f:
        json.dump(sensors_with_both_pressures, f, indent=2, ensure_ascii=False)
    
    print(f"Đã lưu danh sách cảm biến vào: {sensors_file}")
    print("\nBắt đầu xử lý từng cảm biến...")
    
    # Xử lý từng cảm biến bằng doan_forecasting.py
    all_results = {}
    
    for i, sensor_id in enumerate(sensors_with_both_pressures):
        print(f"\n[{i+1}/{len(sensors_with_both_pressures)}] Đang xử lý cảm biến: {sensor_id}")
        
        # Chạy script doan_forecasting.py cho cảm biến hiện tại
        sensor_output_dir = os.path.join(output_dir, sensor_id)
        os.makedirs(sensor_output_dir, exist_ok=True)
        
        cmd = [
            sys.executable,
            "doan_forecasting.py",
            "--sensor_id", sensor_id,
            "--data_path", data_path,
            "--output_dir", sensor_output_dir,
            "--visualize"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Lưu log
            log_file = os.path.join(sensor_output_dir, "processing_log.txt")
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n\n")
                f.write(f"Return code: {result.returncode}\n")
            
            # Ghi kết quả
            all_results[sensor_id] = {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "log_file": log_file
            }
            
            if result.returncode == 0:
                print(f"  ✓ Xử lý thành công cảm biến {sensor_id}")
            else:
                print(f"  ✗ Lỗi khi xử lý cảm biến {sensor_id}, mã lỗi: {result.returncode}")
                print(f"  Xem chi tiết trong: {log_file}")
                
        except Exception as e:
            print(f"  ✗ Lỗi khi xử lý cảm biến {sensor_id}: {str(e)}")
            all_results[sensor_id] = {
                "success": False,
                "error": str(e)
            }
    
    # Lưu kết quả tổng hợp
    summary = {
        "processed_time": datetime.now().isoformat(),
        "data_path": data_path,
        "output_dir": output_dir,
        "total_sensors": len(sensors_with_both_pressures),
        "successful_sensors": sum(1 for r in all_results.values() if r.get("success", False)),
        "failed_sensors": sum(1 for r in all_results.values() if not r.get("success", False)),
        "sensor_results": all_results
    }
    
    summary_file = os.path.join(output_dir, f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nXử lý hoàn tất!")
    print(f"Đã xử lý {summary['successful_sensors']}/{summary['total_sensors']} cảm biến thành công")
    print(f"Báo cáo tổng hợp: {summary_file}")
    
    return 0


if __name__ == "__main__":
    try:
        print("Bắt đầu xử lý các cảm biến có 2 áp suất")
        exit_code = main()
        print(f"Kết thúc với mã thoát: {exit_code}")
        exit(exit_code)
    except Exception as e:
        print(f"Lỗi không xử lý được: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 