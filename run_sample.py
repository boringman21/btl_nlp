#!/usr/bin/env python
"""
Mẫu script chạy phân tích sử dụng package btl_nlp.
Trước khi chạy script, đảm bảo:
1. Bạn đã cài đặt và kích hoạt môi trường ảo
2. Package btl_nlp đã được cài đặt
3. Dữ liệu đã sẵn sàng ở đường dẫn được chỉ định
"""

import os
import matplotlib.pyplot as plt
from btl_nlp.data.data_loader import DataLoader
from btl_nlp.data.data_transform import transform_df, add_derived_metrics
from btl_nlp.utils.visualization import plot_flow_and_pressure, plot_correlation
from btl_nlp.models.fourier import plot_fourier_approximation
from btl_nlp.utils.memory_utils import print_memory_usage, clear_memory

def main():
    # Đường dẫn đến dữ liệu (cần điều chỉnh theo máy của bạn)
    data_path = "./LeakDataset/Logger_Data_2024_Bau_Bang-2"
    
    # Tạo thư mục đầu ra cho các plots
    output_dir = "./output_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Bắt đầu phân tích dữ liệu rò rỉ...")
    print("Đường dẫn dữ liệu:", data_path)
    
    # Theo dõi sử dụng bộ nhớ
    print_memory_usage()
    
    try:
        # Tải dữ liệu
        print("\n1. Đang tải dữ liệu...")
        data_loader = DataLoader(data_path)
        data = data_loader.load_all_data()
        print(f"   - Đã tải xong {len(data)} tập dữ liệu")
        
        # Chuyển đổi dữ liệu
        print("\n2. Đang chuyển đổi dữ liệu...")
        result_df = transform_df(data['merged_data'])
        print(f"   - Dữ liệu sau chuyển đổi có kích thước: {result_df.shape}")
        
        # Tính toán các metrics dẫn xuất
        print("\n3. Đang tính toán các metrics dẫn xuất...")
        result_df = add_derived_metrics(result_df)
        
        # Lấy danh sách sensor IDs
        sensor_ids = data_loader.get_sensor_ids(result_df)
        print(f"   - Tìm thấy {len(sensor_ids)} cảm biến")
        
        if sensor_ids:
            # Chọn một cảm biến để phân tích
            sample_sensor = sensor_ids[0]
            print(f"\n4. Phân tích cảm biến mẫu: {sample_sensor}")
            
            # Vẽ đồ thị dòng chảy và áp suất
            print("   - Đang vẽ đồ thị dòng chảy và áp suất...")
            fig = plot_flow_and_pressure(result_df, sample_sensor, save_dir=output_dir)
            if fig:
                plt.close(fig)  # Đóng hình để tránh hiển thị khi chạy script
            
            # Vẽ ma trận tương quan
            print("   - Đang vẽ ma trận tương quan...")
            fig = plot_correlation(result_df, sample_sensor, save_dir=output_dir)
            if fig:
                plt.close(fig)  # Đóng hình để tránh hiển thị khi chạy script
            
            # Phân tích Fourier
            flow_col = f"{sample_sensor}_Flow"
            if flow_col in result_df.columns:
                print("   - Đang thực hiện phân tích Fourier...")
                fig = plot_fourier_approximation(
                    result_df['Timestamp'],
                    result_df[flow_col].values,
                    'Flow',
                    sample_sensor,
                    save_dir=output_dir,
                    num_terms=10
                )
                if fig:
                    plt.close(fig)  # Đóng hình để tránh hiển thị khi chạy script
        
        print("\nPhân tích hoàn tất. Các đồ thị đã được lưu ở:", output_dir)
    
    except Exception as e:
        print(f"\nLỖI: {str(e)}")
    
    finally:
        # Giải phóng bộ nhớ
        print("\nĐang giải phóng bộ nhớ...")
        clear_memory()

if __name__ == "__main__":
    main() 