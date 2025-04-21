# Refactor Water Leak Project

## Current tasks from user prompt
- Refactor mã nguồn để hoàn chỉnh các công việc hiện đang có
- Cải thiện cấu trúc dự án từ các script rời rạc thành một package đồng nhất

## Plan (simple)
1. Phân tích hiện trạng các script hiện có và cấu trúc package
2. Thống nhất các chức năng chồng chéo và di chuyển code vào package chính
3. Đảm bảo các script có thể hoạt động với cấu trúc mới
4. Sửa lỗi và hoàn thiện các chức năng còn thiếu

## Steps
1. Kiểm tra cấu trúc hiện tại của package water_leakage
2. Xem xét mã nguồn trong các script chính
3. Phân tích các module đang thiếu hoặc cần cải thiện
4. Refactor các chức năng từ script riêng lẻ vào package chính
5. Cập nhật các script để sử dụng package sau khi refactor
6. Kiểm tra và sửa lỗi predict_next_tick.py
7. Hoàn thiện các ứng dụng ưu tiên trong analyze_priority_app.py

## Things done
- Kiểm tra cấu trúc hiện tại của package water_leakage
  - Package đã có cấu trúc cơ bản với các module data, models, utils, apps và tests
  - Module apps đã có priority_applications.py, main.py, và visualize.py
  - Các module data và utils đã có các file cần thiết
  - Module models có fourier.py và time_series.py
- Xem xét mã nguồn trong các script chính
  - apps/main.py thực hiện việc phân tích 4 ứng dụng ưu tiên và có CLI interface
  - apps/priority_applications.py có các hàm phân tích chính
  - apps/visualize.py cung cấp chức năng tạo dashboard và visualize
  - Các script riêng lẻ (run_sample.py, visualize_water_leakage.py, predict_next_tick.py) chưa thống nhất với module apps
- Phân tích các module đang thiếu hoặc cần cải thiện
  - predict_next_tick.py chưa sử dụng đầy đủ các chức năng từ module time_series
  - analyze_priority_app.py quá đơn giản, chỉ gọi đến main.py
  - Chưa có script thống nhất để người dùng sử dụng các chức năng
- Refactor các chức năng từ script riêng lẻ vào package chính
  - Đã tạo water_leakage_analyzer.py làm script thống nhất để gọi các chức năng
  - Script này tích hợp tất cả chức năng từ các script riêng lẻ
  - Sử dụng các module trong package water_leakage
- Cập nhật các script để sử dụng package sau khi refactor
  - Đã cập nhật water_leakage/utils/visualization.py để thêm hàm plot_predictions
  - Đã cập nhật README.md để phản ánh cấu trúc dự án mới và cách sử dụng script thống nhất
- Kiểm tra và sửa lỗi predict_next_tick.py
  - Chức năng dự đoán đã được tích hợp vào script thống nhất water_leakage_analyzer.py
  - Đã sửa lỗi thiếu import pandas trong module visualization.py
- Hoàn thiện các ứng dụng ưu tiên trong analyze_priority_app.py
  - Đã tích hợp analyze_priority_app.py vào script thống nhất water_leakage_analyzer.py

## Things not done yet
- Cần thêm testing cho các script đã refactor 