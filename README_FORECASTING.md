# Hướng dẫn sử dụng Công cụ Dự đoán và Phát hiện Bất thường

Tài liệu này hướng dẫn cách sử dụng công cụ `doan_forecasting.py` để dự đoán lưu lượng nước và phát hiện bất thường theo yêu cầu của giảng viên.

## Mục tiêu

Công cụ được triển khai dựa trên các yêu cầu cụ thể:

1. Dự đoán dữ liệu Flow theo khung thời gian 6h với output dự đoán là tick 15p tiếp theo
2. Phát hiện điểm bất thường bằng cách so sánh dữ liệu thực tế và dữ liệu dự đoán. Tính standard error của sai số dự đoán trên tập dữ liệu lịch sử. Nếu độ lệch giữa giá trị dự đoán và giá trị thực tế tại một thời điểm lớn hơn 3 lần standard error, thì điểm đó được coi là bất thường.
3. Dự đoán Flow không chỉ tận dụng feature Flow mà tận dụng thêm chênh lệch áp suất với điểm có 2 điểm đo pressure.
4. Dự đoán Flow cho các điểm đo không có 2 pressure mà chỉ có 1 pressure.

## Cài đặt

1. Đảm bảo Python 3.6+ đã được cài đặt
2. Cài đặt các gói phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

## Cách sử dụng

### Cú pháp lệnh

```
python doan_forecasting.py [options]
```

### Các tùy chọn

- `--data_path DATA_PATH`: Đường dẫn đến thư mục dữ liệu logger (mặc định: "./LeakDataset/Logger_Data_2024_Bau_Bang-2")
- `--output_dir OUTPUT_DIR`: Thư mục lưu kết quả dự đoán (mặc định: "./prediction_results")  
- `--sensor_id SENSOR_ID`: ID cảm biến cụ thể để dự đoán. Nếu không cung cấp, sẽ dự đoán cho tất cả cảm biến có dữ liệu đầy đủ
- `--model_dir MODEL_DIR`: Thư mục lưu/tải mô hình đã huấn luyện (mặc định: "./models")
- `--visualize`: Tạo trực quan hóa dự đoán và phát hiện bất thường
- `--force_retrain`: Buộc huấn luyện lại mô hình ngay cả khi chúng đã tồn tại

### Ví dụ sử dụng

1. Dự đoán tất cả các cảm biến và tạo trực quan hóa:
   ```
   python doan_forecasting.py --visualize
   ```

2. Dự đoán một cảm biến cụ thể:
   ```
   python doan_forecasting.py --sensor_id 841210607378 --visualize
   ```

3. Sử dụng bộ dữ liệu khác:
   ```
   python doan_forecasting.py --data_path ./LeakDataset/Other_Logger_Data --output_dir ./other_results
   ```

## Kết quả

Các kết quả sẽ được lưu trong thư mục output (mặc định là "./prediction_results"):

1. **Dự đoán**: File JSON chứa các dự đoán cho tick 15 phút tiếp theo
   - `enhanced_predictions_YYYYMMDD_HHMMSS.json`

2. **Phát hiện bất thường**: File JSON chứa các điểm bất thường được phát hiện
   - `anomalies_YYYYMMDD_HHMMSS.json`
   
3. **Báo cáo tổng hợp**: File JSON chứa thông tin tổng hợp về dự đoán và bất thường
   - `report_YYYYMMDD_HHMMSS.json`

4. **Trực quan hóa**: Các file hình ảnh PNG cho mỗi cảm biến (nếu sử dụng tùy chọn `--visualize`)
   - `SENSOR_ID_prediction_anomalies.png`

## Kiến trúc

### Các lớp chính

1. **EnhancedTimeSeriesPredictor**: Mở rộng từ TimeSeriesPredictor, thêm khả năng tận dụng chênh lệch áp suất làm feature để dự đoán

2. **Chức năng phát hiện bất thường**: Sử dụng phương pháp standard error để phát hiện các điểm bất thường

### Quy trình xử lý dữ liệu

1. **Tải dữ liệu**: Đọc dữ liệu từ thư mục logger
2. **Chuyển đổi dữ liệu**: Chuẩn hóa dữ liệu để sử dụng
3. **Xác định cảm biến**: Phân loại cảm biến theo số lượng điểm đo áp suất
4. **Dự đoán Flow**: 
   - Tận dụng dữ liệu 6 giờ để dự đoán tick 15 phút tiếp theo
   - Sử dụng chênh lệch áp suất khi có cả 2 điểm đo
   - Sử dụng mô hình tương quan cho cảm biến chỉ có 1 điểm đo áp suất
5. **Phát hiện bất thường**: 
   - Tính standard error từ sai số dự đoán trên dữ liệu lịch sử
   - Đánh dấu các điểm có độ lệch > 3 lần standard error là bất thường
6. **Trực quan hóa**: 
   - Biểu diễn dữ liệu thực tế, dự đoán và các điểm bất thường
   - Hiển thị ngưỡng (threshold) phát hiện bất thường
   - Biểu đồ error sigma cho thấy mức độ bất thường

## Tài liệu tham khảo

1. [Dự báo chuỗi thời gian sử dụng LSTM/RNN](https://www.tensorflow.org/tutorials/structured_data/time_series)
2. [Phát hiện dị thường với standard error](https://en.wikipedia.org/wiki/Standard_error)
3. [Các kỹ thuật phát hiện rò rỉ nước qua phân tích áp suất](https://www.mdpi.com/2073-4441/12/4/1096) 