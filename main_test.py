from water_leakage.data.data_loader import DataLoader
from water_leakage.data.data_transform import transform_df, add_derived_metrics, identify_potential_leaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import os
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.patches import Patch

# Tạo thư mục để lưu biểu đồ
os.makedirs('./plots', exist_ok=True)

data_path = './LeakDataset/Logger_Data_2024_Bau_Bang-2'
# Load data
print("Loading data...")
data_loader = DataLoader(data_path)
data = data_loader.load_all_data()

# Transform data
print("Transforming data...")
sms_numbers = [
    84210802044,
    84797805118,
    841212383325,
    841211914190,
    841210802048,
    841210802047,
    841210620665,
    841210607378,
    840786560116,
    8401210607558
]



def get_clean_df(result_df, missing_by_date):
    clean_df = result_df.copy().reset_index()
    problematic_dates = missing_by_date.index
    for date in problematic_dates:
        clean_df = clean_df[~(clean_df['Timestamp'].dt.date == date)]
    return clean_df

def get_missing_by_date(result_df, full_range):
    missing = full_range.difference(result_df.index)
    if len(missing) == 0:
        return pd.Series(dtype=int)
    missing_by_date = missing.to_frame(index=False, name='timestamp').groupby(missing.date).size()
    return missing_by_date

def plot_timeseries_and_filtered_days(clean_df, missing_by_date, sms):
    # Đảm bảo clean_df có index là kiểu datetime
    if not isinstance(clean_df.index, pd.DatetimeIndex):
        if 'Timestamp' in clean_df.columns:
            clean_df = clean_df.set_index('Timestamp').sort_index()
        else:
            # Nếu không có cột Timestamp, tạo index datetime mới
            clean_df.index = pd.to_datetime(clean_df.index)
            
    # Xác định tên cột phù hợp cho các metrics
    flow_col = next((col for col in clean_df.columns if 'flow' in col.lower()), None)
    pressure1_col = next((col for col in clean_df.columns if 'pressure_1' in col.lower() or 'pressure1' in col.lower() or 'p1' in col.lower()), None)
    pressure2_col = next((col for col in clean_df.columns if 'pressure_2' in col.lower() or 'pressure2' in col.lower() or 'p2' in col.lower()), None)
    
    # Tính toán chênh lệch áp suất nếu không có cột chênh lệch áp suất
    if 'pressure_diff' not in clean_df.columns:
        if pressure1_col and pressure2_col:
            clean_df['pressure_diff'] = clean_df[pressure1_col] - clean_df[pressure2_col]
            pressure_diff_col = 'pressure_diff'
        else:
            pressure_diff_col = None
    else:
        pressure_diff_col = 'pressure_diff'
    
    # Tạo một biểu đồ kết hợp với 5 subplot: 4 cho các metrics và 1 cho thông tin ngày đã lọc
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.5], hspace=0)
    # Thêm khoảng cách giữa subplot thứ 4 và 5
    gs.update(hspace=0, top=0.95)
    fig.suptitle(f'Dữ liệu cảm biến và thông tin lọc cho SMS {sms}', fontsize=18, y=0.98)

    # Xác định phạm vi thời gian cho tất cả các biểu đồ để đảm bảo căn chỉnh
    min_date = clean_df.index.min()
    max_date = clean_df.index.max()
    min_date = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
    max_date = (max_date + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(microseconds=1)

    # Tạo month_labels cho tất cả các plot
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    tick_positions = range(len(date_range))
    month_labels = [d.strftime('%m-%Y') if d.day == 1 else '' for d in date_range]

    # Subplot 1: Flow
    ax1 = fig.add_subplot(gs[0])
    if flow_col:
        clean_df[flow_col].plot(ax=ax1, color='blue', marker='.', markersize=2, linestyle='-', linewidth=1)
        ax1.set_ylabel(f'{flow_col} (m³/h)', fontweight='bold')
        # ax1.set_title('Lưu lượng nước', fontsize=12)
        ax1.grid(True)
    else:
        ax1.set_title('Không có dữ liệu lưu lượng', fontsize=12)
    ax1.set_xlim(min_date, max_date)
    # Thiết lập định dạng ngày tháng cho trục x của ax1
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())

    # Subplot 2: Pressure_1
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if pressure1_col:
        clean_df[pressure1_col].plot(ax=ax2, color='green', marker='.', markersize=2, linestyle='-', linewidth=1)
        ax2.set_ylabel(f'{pressure1_col} (bar)', fontweight='bold')
        # ax2.set_title('Áp suất 1', fontsize=12)
        ax2.grid(True)
    else:
        ax2.set_title('Không có dữ liệu áp suất 1', fontsize=12)
    # Thiết lập định dạng ngày tháng cho trục x của ax2
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    # Subplot 3: Pressure_2
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if pressure2_col:
        clean_df[pressure2_col].plot(ax=ax3, color='red', marker='.', markersize=2, linestyle='-', linewidth=1)
        ax3.set_ylabel(f'{pressure2_col} (bar)', fontweight='bold')
        # ax3.set_title('Áp suất 2', fontsize=12)
        ax3.grid(True)
    else:
        ax3.set_title('Không có dữ liệu áp suất 2', fontsize=12)
    # Thiết lập định dạng ngày tháng cho trục x của ax3
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())

    # Subplot 4: Pressure_Diff
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if pressure_diff_col:
        clean_df[pressure_diff_col].plot(ax=ax4, color='purple', marker='.', markersize=2, linestyle='-', linewidth=1)
        ax4.set_ylabel(f'{pressure_diff_col} (bar)', fontweight='bold')
        # ax4.set_title('Chênh lệch áp suất', fontsize=12)
        ax4.grid(True)
    else:
        ax4.set_title('Không có dữ liệu chênh lệch áp suất', fontsize=12)
    # Thiết lập định dạng ngày tháng cho trục x của ax4
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    # Đặt rotation=0 cho nhãn trục x của ax4
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0, ha='center')
    # ax4.set_xlabel('Thời gian', fontsize=12, fontweight='bold')

    # Định dạng trục x cho các subplot metrics
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, axis='x', linestyle='-', alpha=0.3, color='black')
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        # Định dạng tick và nhãn trục x đã được thiết lập riêng cho từng ax ở trên
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        if ax != ax4:
            plt.setp(ax.get_xticklabels(), visible=False)

    # Subplot 5: Heatmap thông tin về các ngày đã lọc bỏ
    ax5 = fig.add_subplot(gs[4])
    ax5.set_title('Các ngày bị thiếu dữ liệu (đã bị lọc bỏ)', fontsize=12)
    start_date = min_date.date()
    end_date = max_date.date()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    date_status = pd.DataFrame(index=all_dates)
    date_status['date'] = date_status.index.date
    date_status['status'] = 'valid'
    if not missing_by_date.empty:
        for d in missing_by_date.index:
            mask = (date_status['date'] == d)
            date_status.loc[mask, 'status'] = 'filtered'
    status_colors = {'valid': 0, 'filtered': 1}
    date_status['color_code'] = date_status['status'].map(status_colors)
    heatmap_data = date_status['color_code'].values.reshape(1, -1)
    cmap = plt.cm.colors.ListedColormap(['#4CAF50', '#F44336'])
    
    # Tạo xticks và xticklabels phù hợp để tránh cảnh báo
    xticks = np.arange(len(all_dates))
    xticklabels = [d.strftime('%m-%Y') if d.day == 1 else '' for d in all_dates]
    
    hm = sns.heatmap(heatmap_data, ax=ax5, cmap=cmap, cbar=False, 
                     xticklabels=[], yticklabels=[''])
    ax5.set_xlim(0, len(all_dates))
    for i in range(1, len(all_dates)):
        ax5.axvline(i, color='white', lw=0.5)
    
    # Thiết lập ticks và ticklabels theo cách đúng
    ax5.set_xticks([i for i, label in enumerate(xticklabels) if label])
    ax5.set_xticklabels([label for label in xticklabels if label], ha='right')
    
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Ngày có dữ liệu hợp lệ'),
        Patch(facecolor='#F44336', label='Ngày bị lọc bỏ do thiếu dữ liệu')
    ]
    ax5.legend(handles=legend_elements, loc='upper right')
    
    # Điều chỉnh layout và khoảng cách giữa các subplot
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.1, top=0.95, wspace=0)
    
    # Tăng khoảng cách riêng giữa ax4 và ax5
    pos4 = ax4.get_position()
    pos5 = ax5.get_position()
    pos5.y0 = pos4.y0 - 0.15  # Tăng khoảng cách thêm 0.15
    pos5.y1 = pos4.y0 - 0.05
    ax5.set_position(pos5)
    
    plt.savefig(f'./plots/timeseries_combined_{sms}.png', dpi=300, bbox_inches='tight')
    print(f"Đã lưu biểu đồ kết hợp vào ./plots/timeseries_combined_{sms}.png")
    plt.close(fig)


sms_tick_15m = [
    # 84797805118, # ko có pressure 1
    841210802048,
    # 841210802047, # dữ liệu thiếu quá nhiều
    841210620665, # ko có dữ liệu flow
    841210607378, # ko có dữ liệu flow
    
    840786560116, # có flow nhưng thiếu pressure 1 và 2, chỉ có thể dùng dể flow train cho flow
    841211914190, # dữ liệu có thể dùng để train
    8401210607558, # đủ dữ liệu dể train
]

sms_tick_5m = [
    841212383325, # dữ liệu thiếu nhiều và bắt đầu từ tháng 6, ko có dữ liệu flow
]

sms_have_full_data = [
    8401210607558,
]



full_range_dataset = pd.date_range(
    start='2024-01-01T00:00:00.000Z',
    end='2024-12-31T23:45:00.000Z',
    freq='15min'
)

def clean_data(merged_df, sms):
    sms_data = merged_df[merged_df['smsNumber'] == sms]
    # lưu sms data vào file csv: output/sms_data_{sms}.csv
    sms_data.to_csv(f"./output/sms_data_{sms}.csv", index=False)
    
    sms_df = transform_df(sms_data)
    # lưu transformed data vào file csv: output/transformed_data_{sms}.csv
    sms_df.to_csv(f"./output/transformed_data_{sms}.csv", index=False)

    # add if to check datatype before ép kiểu
    if sms_df['Timestamp'].dtype != 'datetime64[ns]':
        sms_df['Timestamp'] = pd.to_datetime(sms_df['Timestamp'])
        
    # check if Timestamp is index
    if sms_df.index.name != 'Timestamp':
        sms_df = sms_df.set_index('Timestamp').sort_index()
    
    # kiểm tra các dữ liệu missing
    missing_by_date = get_missing_by_date(sms_df, full_range_dataset)

    # lưu missing_by_date vào file csv: output/missing_by_date_{sms}.csv
    missing_by_date.to_csv(f"./output/missing_by_date_{sms}.csv", index=True, header=True)
    
    # lấy clean_df
    clean_df = get_clean_df(sms_df, missing_by_date)
    # lưu clean_df vào file csv: output/clean_data_{sms}.csv
    clean_df.to_csv(f"./output/clean_data_{sms}.csv", index=False)
    
    # Đảm bảo clean_df có index là kiểu datetime trước khi trả về
    if 'Timestamp' in clean_df.columns:
        clean_df = clean_df.set_index('Timestamp').sort_index()

    return clean_df, missing_by_date

for sms in sms_tick_5m:
    cleaned_df, missing_by_date = clean_data(data['merged_data'], sms)
    plot_timeseries_and_filtered_days(cleaned_df, missing_by_date, sms)

