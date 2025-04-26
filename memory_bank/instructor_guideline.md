Hướng dẫn của giảng viên hướng dẫn:
1. Hãy dữ đoán dữ liệu Flow theo khung thời gian 6h với output dự đoán là tick 15p tiếp theo
2. Phát hiện điểm bất thường bằng cách so sánh dữ liệu thực tế và dữ liệu dự đoán ở trên. Tính standard error của sai số dự đoán trên tập dữ liệu lịch sử. Nếu độ lệch giữa giá trị dự đoán và giá trị thực tế tại một thời điểm lớn hơn 3 lần standard error, thì điểm đó được coi là bất thường.
2.1. Sử dụng Isolation Tree để nhận biết điểm dị thường.

3. Hãy dự đoán Flow không chỉ tận dụng feature Flow mà tận dụng thêm chênh lệch áp suất với điểm có 2 điểm đo pressure.sư
4. Dự đoán Flow cho các điểm đo không có 2 pressure mà chỉ có 1 pressure.
5. Fill các giá trị của pressures, flow bằng các MA 4 điểm có thể lấy trước sau, hoặc chỉ trước.
6. Dùng thêm GRU để dự đoán.
7. Xây dụng model chung cho tất cả dữ liệu so sánh với model riêng của từng trạm.
8. Nếu có dữ liệu network thì cân nhắc dùng thêm
