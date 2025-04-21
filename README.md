# Water Leakage Analysis Package

Phân tích dữ liệu rò rỉ nước từ các cảm biến áp suất và lưu lượng.

## Mô tả

Package này phân tích dữ liệu từ các cảm biến theo dõi hệ thống nước để phát hiện rò rỉ, dự đoán mẫu tiêu thụ, cảnh báo sớm và phát hiện bất thường.

## Cấu trúc dự án

```
btl_nlp/
│
├── water_leakage/             # Package chính
│   ├── __init__.py
│   │
│   ├── data/                  # Quản lý dữ liệu
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Tải và tiền xử lý dữ liệu
│   │   └── data_transform.py  # Chuyển đổi dữ liệu
│   │
│   ├── utils/                 # Tiện ích
│   │   ├── __init__.py
│   │   ├── memory_utils.py    # Tiện ích theo dõi bộ nhớ
│   │   └── visualization.py   # Trực quan hóa dữ liệu
│   │
│   ├── models/                # Mô hình phân tích
│   │   ├── __init__.py
│   │   ├── fourier.py         # Xấp xỉ Fourier
│   │   └── time_series.py     # Dự đoán chuỗi thời gian
│   │
│   ├── apps/                  # Ứng dụng cụ thể
│   │   ├── __init__.py
│   │   ├── priority_applications.py  # Các ứng dụng ưu tiên
│   │   ├── main.py            # CLI và điều phối
│   │   └── visualize.py       # Dashboard và trực quan hóa
│   │
│   └── tests/                 # Unit tests
│       └── __init__.py
│
├── water_leakage_analyzer.py  # Script thống nhất
├── analyze_priority_app.py    # Script chạy ứng dụng ưu tiên
├── run_sample.py              # Script mẫu phân tích cơ bản
├── visualize_water_leakage.py # Script tạo biểu đồ
├── predict_next_tick.py       # Script dự đoán giá trị tiếp theo
├── README.md                  # Tài liệu dự án
└── requirements.txt           # Các phụ thuộc
```

## Các tính năng chính

1. **Phát hiện rò rỉ**: Phân tích sự khác biệt về áp suất và mẫu dòng chảy bất thường
2. **Dự đoán mẫu tiêu thụ nước**: Phân tích dữ liệu lịch sử để dự báo nhu cầu tương lai
3. **Hệ thống cảnh báo sớm**: Phát hiện những thay đổi bất thường trước khi xảy ra rò rỉ lớn
4. **Phát hiện và dự đoán bất thường**: Xác định các mẫu bất thường

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd btl_nlp
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Cài đặt package:
```bash
pip install -e .
```

## Sử dụng

### Script thống nhất

Script `water_leakage_analyzer.py` cung cấp giao diện thống nhất cho tất cả chức năng:

```bash
# Phân tích cơ bản
python water_leakage_analyzer.py --mode basic-analysis --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2

# Phân tích ứng dụng ưu tiên
python water_leakage_analyzer.py --mode priority-apps --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2

# Dự đoán giá trị tiếp theo
python water_leakage_analyzer.py --mode prediction --data_path ./LeakDataset/Logger_Data_2024_Bau_Bang-2

# Trực quan hóa kết quả đã lưu
python water_leakage_analyzer.py --mode visualize --results_file ./analysis_results/results/analysis_results.json
```

### Các tùy chọn chung

- `--data_path`: Đường dẫn thư mục dữ liệu
- `--output_dir`: Thư mục lưu kết quả
- `--sensor_id`: ID cảm biến cụ thể để phân tích

### Các script riêng lẻ

Các script riêng lẻ vẫn hoạt động được, nhưng khuyến nghị sử dụng script thống nhất:

- `run_sample.py`: Chạy phân tích cơ bản
- `analyze_priority_app.py`: Chạy phân tích ứng dụng ưu tiên
- `predict_next_tick.py`: Dự đoán giá trị tiếp theo
- `visualize_water_leakage.py`: Trực quan hóa kết quả

## Đặc điểm trực quan hóa

Package bao gồm nhiều loại trực quan hóa:

1. **Biểu đồ cơ bản**: Dòng chảy, áp suất và tương quan
2. **Bảng điều khiển tích hợp**: Kết hợp kết quả từ tất cả ứng dụng ưu tiên
3. **Phân tích so sánh**: Biểu đồ so sánh giữa các cảm biến

## Tài liệu tham khảo

Để biết thêm chi tiết về các tính năng trực quan hóa, hãy xem thư mục `report`. 