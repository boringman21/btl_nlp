#!/usr/bin/env python3
"""
Dự đoán Flow và phát hiện bất thường theo yêu cầu giảng viên:
1. Dự đoán dữ liệu Flow theo khung thời gian 6h với output dự đoán là tick 15p tiếp theo
2. Nhận đoán điểm bất thường 
3. Dự đoán Flow tận dụng cả chênh lệch áp suất với điểm có 2 điểm đo pressure
4. Dự đoán Flow cho các điểm đo không có 2 pressure mà chỉ có 1 pressure
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import sys

from water_leakage.data.data_loader import load_data
from water_leakage.data.data_transform import transform_df
from water_leakage.models.time_series import TimeSeriesPredictor

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Phan tich tham so dong lenh."""
    parser = argparse.ArgumentParser(
        description="Du doan Flow va phat hien bat thuong theo yeu cau giang vien"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./LeakDataset/Logger_Data_2024_Bau_Bang-2",
        help="Duong dan den thu muc du lieu logger"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./prediction_results",
        help="Thu muc luu ket qua du doan"
    )
    
    parser.add_argument(
        "--sensor_id", 
        type=str,
        help="ID cam bien cu the de du doan. Neu khong cung cap, se du doan cho tat ca cam bien co du lieu day du."
    )
    
    parser.add_argument(
        "--model_dir", 
        type=str,
        default="./models",
        help="Thu muc luu/tai mo hinh da huan luyen"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Tao truc quan hoa du doan"
    )
    
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Buoc huan luyen lai mo hinh ngay ca khi chung da ton tai"
    )
    
    return parser.parse_args()


class EnhancedTimeSeriesPredictor(TimeSeriesPredictor):
    """
    Bộ dự đoán chuỗi thời gian nâng cao, mở rộng từ TimeSeriesPredictor cơ bản.
    Thêm khả năng tận dụng chênh lệch áp suất và xử lý điểm chỉ có 1 giá trị áp suất.
    """
    
    def __init__(self, window_size=24, use_pressure_diff=True):
        """
        Khởi tạo bộ dự đoán chuỗi thời gian nâng cao.
        
        Args:
            window_size (int): Số bước thời gian sử dụng để dự đoán (mặc định: 24, tương đương 6 giờ với các khoảng 15 phút)
            use_pressure_diff (bool): Có sử dụng chênh lệch áp suất làm feature không
        """
        super().__init__(window_size=window_size)
        self.use_pressure_diff = use_pressure_diff
    
    def preprocess_data(self, df):
        """
        Tiền xử lý dữ liệu cho dự đoán chuỗi thời gian.
        Thêm tính năng chênh lệch áp suất nếu có.
        
        Args:
            df (pd.DataFrame): DataFrame đầu vào với dữ liệu chuỗi thời gian
            
        Returns:
            np.ndarray: Dữ liệu đã tiền xử lý
        """
        # Đảm bảo timestamp là index
        if 'Timestamp' in df.columns:
            df = df.set_index('Timestamp')
        
        # Nội suy các giá trị thiếu
        df = df.interpolate(method='time')
        
        # Thêm tính năng thời gian
        if df.index.dtype.kind == 'M':  # Kiểm tra nếu index là datetime
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
        
        # Thêm tính năng chênh lệch áp suất nếu có Pressure_1 và Pressure_2
        if self.use_pressure_diff:
            pressure_cols = [col for col in df.columns if 'Pressure_1' in col or 'Pressure_2' in col]
            sensor_ids = set()
            
            for col in pressure_cols:
                parts = col.split('_')
                if len(parts) >= 2:
                    sensor_ids.add(parts[0])
            
            for sensor_id in sensor_ids:
                p1_col = f"{sensor_id}_Pressure_1"
                p2_col = f"{sensor_id}_Pressure_2"
                
                if p1_col in df.columns and p2_col in df.columns:
                    df[f"{sensor_id}_Pressure_Diff"] = df[p1_col] - df[p2_col]
        
        # Scale dữ liệu
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df)
        
        return scaled_data


def detect_anomalies(df, sensor_id, feature_cols, predictions=None, window_size=24):
    """
    Phat hien diem bat thuong bang cach so sanh du lieu thuc te va du lieu du doan.
    Tinh standard error cua sai so du doan tren tap du lieu lich su.
    Neu do lech giua gia tri du doan va gia tri thuc te tai mot thoi diem lon hon 3 lan standard error,
    thi diem do duoc coi la bat thuong.
    
    Args:
        df (pd.DataFrame): DataFrame du lieu
        sensor_id (str): ID cam bien de phat hien bat thuong
        feature_cols (list): Danh sach cot tinh nang su dung
        predictions (dict, optional): Du doan da co san
        window_size (int): Kich thuoc cua so du doan (24 = 6h voi khoang 15 phut)
    
    Returns:
        pd.DataFrame: DataFrame voi cot 'anomaly' danh dau cac diem bat thuong
    """
    # Loc DataFrame cho cam bien cu the va cac tinh nang
    sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols if f"{sensor_id}_{col}" in df.columns]
    
    # Kiem tra du cot
    if not sensor_cols:
        logger.error(f"Khong co cot hop le cho cam bien {sensor_id}")
        return None
    
    # Trich xuat du lieu
    sensor_df = df[['Timestamp'] + sensor_cols].copy()
    
    # Chuyen doi timestamp thanh datetime neu chua
    sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
    
    # Them chenh lech ap suat neu co du du lieu
    p1_col = f"{sensor_id}_Pressure_1"
    p2_col = f"{sensor_id}_Pressure_2"
    
    if p1_col in sensor_df.columns and p2_col in sensor_df.columns:
        sensor_df[f"{sensor_id}_Pressure_Diff"] = sensor_df[p1_col] - sensor_df[p2_col]
        sensor_cols.append(f"{sensor_id}_Pressure_Diff")
    
    # Tao mo hinh du doan
    model_cols = sensor_cols.copy()
    flow_col = f"{sensor_id}_Flow"
    
    if flow_col not in model_cols:
        logger.error(f"Khong tim thay cot Flow cho cam bien {sensor_id}")
        return None
    
    # Dat index cho DataFrame
    sensor_df.set_index('Timestamp', inplace=True)
    
    # Tao cac du doan tren tap du lieu lich su
    y_true = []
    y_pred = []
    
    # Su dung EnhancedTimeSeriesPredictor de tao cac du doan
    if predictions is None:
        predictor = EnhancedTimeSeriesPredictor(window_size=window_size, use_pressure_diff=True)
        
        # Day du lieu de tao du doan
        for i in range(window_size, len(sensor_df) - 1):
            # Lay cua so du lieu
            train_window = sensor_df.iloc[i - window_size:i]
            
            try:
                # Huan luyen mo hinh
                predictor.fit(train_window)
                
                # Du doan gia tri tiep theo
                prediction = predictor.predict_next(train_window)
                
                # Luu gia tri thuc te va du doan
                actual_flow = sensor_df[flow_col].iloc[i]
                predicted_flow = prediction[0]  # Gia tri Flow la phan tu dau tien
                
                y_true.append(actual_flow)
                y_pred.append(predicted_flow)
            except Exception as e:
                logger.warning(f"Loi khi du doan tai thoi diem {i}: {str(e)}")
    
    # Chuyen y_true va y_pred thanh numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Tinh sai so
    errors = y_true - y_pred
    
    # Tinh standard error cua sai so
    se = np.std(errors, ddof=1)
    
    # Dat nguong phat hien bat thuong la 3 lan standard error
    threshold = 3 * se
    
    # Phat hien bat thuong
    anomalies = np.abs(errors) > threshold
    
    # Tao DataFrame ket qua
    result_df = df[['Timestamp'] + sensor_cols].copy()
    result_df['predicted_flow'] = np.nan
    result_df['anomaly'] = 0
    
    # Them cac du doan va danh dau bat thuong
    start_idx = window_size
    end_idx = start_idx + len(y_pred)
    
    for i, (idx, row) in enumerate(result_df.iloc[start_idx:end_idx].iterrows()):
        if i < len(y_pred):
            result_df.loc[idx, 'predicted_flow'] = y_pred[i]
            result_df.loc[idx, 'anomaly'] = int(anomalies[i])
    
    # Tinh do lech va nguong
    result_df['error'] = result_df[flow_col] - result_df['predicted_flow']
    result_df['threshold'] = threshold
    
    # Tinh error sigma (so lan standard error)
    result_df['error_sigma'] = np.abs(result_df['error']) / se
    
    logger.info(f"Standard error cua sai so du doan: {se:.4f}")
    logger.info(f"Nguong phat hien bat thuong (3*SE): {threshold:.4f}")
    
    return result_df


def predict_with_one_pressure(df, sensor_id, feature_cols, similar_sensors=None, model_dir=None):
    """
    Dự đoán dữ liệu Flow cho các điểm chỉ có 1 điểm đo áp suất.
    Sử dụng các cảm biến tương tự làm mô hình tham chiếu nếu có.
    
    Args:
        df (pd.DataFrame): DataFrame dữ liệu
        sensor_id (str): ID cảm biến để dự đoán
        feature_cols (list): Danh sách cột tính năng sử dụng
        similar_sensors (list): Danh sách cảm biến tương tự (có cả 2 điểm đo áp suất)
        model_dir (str): Thư mục mô hình để tải mô hình đã huấn luyện
        
    Returns:
        dict: Dự đoán cho tất cả các tính năng
    """
    # Lọc DataFrame cho cảm biến cụ thể
    available_cols = [col for col in df.columns if col.startswith(f"{sensor_id}_")]
    
    if not available_cols:
        logger.error(f"Không tìm thấy dữ liệu cho cảm biến {sensor_id}")
        return None
    
    # Xác định loại cảm biến (xem có bao nhiêu điểm đo áp suất)
    has_pressure_1 = f"{sensor_id}_Pressure_1" in df.columns
    has_pressure_2 = f"{sensor_id}_Pressure_2" in df.columns
    
    # Nếu có cả 2 điểm đo áp suất, sử dụng phương pháp thông thường
    if has_pressure_1 and has_pressure_2:
        try:
            # Tạo bộ dự đoán nâng cao
            predictor = EnhancedTimeSeriesPredictor(window_size=24, use_pressure_diff=True)
            
            # Lọc DataFrame
            sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols if f"{sensor_id}_{col}" in df.columns]
            sensor_df = df[['Timestamp'] + sensor_cols].copy()
            
            # Chuyển đổi timestamp thành datetime
            sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
            sensor_df = sensor_df.set_index('Timestamp')
            
            # Huấn luyện mô hình
            predictor.fit(sensor_df)
            
            # Dự đoán giá trị tiếp theo
            prediction = predictor.predict_next(sensor_df)
            
            # Chuyển kết quả thành dict
            result = {}
            for i, col in enumerate(feature_cols):
                if i < len(prediction):
                    result[col] = float(prediction[i])
            
            return result
        except Exception as e:
            logger.error(f"Lỗi dự đoán cho cảm biến {sensor_id}: {str(e)}")
            return None
    
    # Nếu chỉ có 1 điểm đo áp suất, sử dụng phương pháp điều chỉnh
    else:
        # Xác định cảm biến tương tự có cả 2 điểm đo áp suất
        if not similar_sensors:
            # Tìm tất cả cảm biến có cả 2 điểm đo áp suất
            all_cols = df.columns
            potential_similar_sensors = set()
            
            for col in all_cols:
                if "_Pressure_1" in col or "_Pressure_2" in col:
                    parts = col.split("_")
                    potential_similar_sensors.add(parts[0])
            
            similar_sensors = []
            for s in potential_similar_sensors:
                if f"{s}_Pressure_1" in all_cols and f"{s}_Pressure_2" in all_cols and f"{s}_Flow" in all_cols:
                    similar_sensors.append(s)
        
        if not similar_sensors:
            logger.error("Không tìm thấy cảm biến tương tự nào có cả 2 điểm đo áp suất")
            return None
        
        # Tạo mô hình dự đoán sử dụng dữ liệu từ cảm biến tương tự
        try:
            # Kết hợp dữ liệu từ cảm biến hiện tại và cảm biến tương tự
            current_cols = [col for col in df.columns if col.startswith(f"{sensor_id}_")]
            combined_df = df[['Timestamp'] + current_cols].copy()
            combined_df.set_index('Timestamp', inplace=True)
            
            # Thêm dữ liệu từ cảm biến tương tự đầu tiên như các tính năng bổ sung
            similar_sensor = similar_sensors[0]
            similar_cols = [col for col in df.columns if col.startswith(f"{similar_sensor}_")]
            
            for col in similar_cols:
                feature_name = col.split("_", 1)[1]  # Lấy phần sau sensor_id
                combined_df[f"similar_{feature_name}"] = df[col].values
            
            # Tạo tính năng chênh lệch áp suất từ cảm biến tương tự
            if f"similar_Pressure_1" in combined_df.columns and f"similar_Pressure_2" in combined_df.columns:
                combined_df["similar_Pressure_Diff"] = combined_df["similar_Pressure_1"] - combined_df["similar_Pressure_2"]
            
            # Huấn luyện mô hình với dữ liệu kết hợp
            predictor = TimeSeriesPredictor(window_size=24)
            predictor.fit(combined_df)
            
            # Dự đoán giá trị tiếp theo
            prediction = predictor.predict_next(combined_df)
            
            # Lấy kết quả dự đoán chỉ cho các tính năng của cảm biến hiện tại
            result = {}
            for col in feature_cols:
                if f"{sensor_id}_{col}" in combined_df.columns:
                    idx = list(combined_df.columns).index(f"{sensor_id}_{col}")
                    if idx < len(prediction):
                        result[col] = float(prediction[idx])
            
            return result
        except Exception as e:
            logger.error(f"Lỗi dự đoán cho cảm biến {sensor_id} chỉ có 1 điểm đo áp suất: {str(e)}")
            return None


def plot_predictions_with_anomalies(df, anomaly_df, prediction, sensor_id, next_timestamp, output_path):
    """
    Truc quan hoa du doan va cac diem bat thuong.
    
    Args:
        df (pd.DataFrame): DataFrame goc
        anomaly_df (pd.DataFrame): DataFrame co danh dau diem bat thuong
        prediction (dict): Du doan gia tri tiep theo
        sensor_id (str): ID cam bien
        next_timestamp (datetime): Thoi diem tiep theo de du doan
        output_path (str): Duong dan de luu bieu do
    """
    try:
        # Chuan bi du lieu cho truc quan hoa
        flow_col = f"{sensor_id}_Flow"
        pressure1_col = f"{sensor_id}_Pressure_1"
        pressure2_col = f"{sensor_id}_Pressure_2"
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
        
        # Ve dong chay (Flow)
        if flow_col in df.columns:
            axs[0].plot(df['Timestamp'], df[flow_col], 'b-', label='Thuc te')
            
            # Ve du doan lich su (neu co)
            if 'predicted_flow' in anomaly_df.columns:
                valid_mask = ~anomaly_df['predicted_flow'].isna()
                axs[0].plot(anomaly_df.loc[valid_mask, 'Timestamp'], 
                           anomaly_df.loc[valid_mask, 'predicted_flow'], 
                           'g--', label='Du doan lich su')
            
            # Ve diem du doan moi
            if 'Flow' in prediction:
                axs[0].scatter([next_timestamp], [prediction['Flow']], color='red', s=100, label='Du doan moi')
            
            # Danh dau diem bat thuong
            if 'anomaly' in anomaly_df.columns:
                anomaly_points = anomaly_df[anomaly_df['anomaly'] == 1]
                if not anomaly_points.empty:
                    axs[0].scatter(anomaly_points['Timestamp'], anomaly_points[flow_col], 
                                color='orange', s=80, marker='*', label='Bat thuong')
            
            # Ve duong nguong
            if 'error' in anomaly_df.columns and 'threshold' in anomaly_df.columns:
                threshold = anomaly_df['threshold'].iloc[0]
                axs[0].fill_between(df['Timestamp'], 
                                  df[flow_col] - threshold, 
                                  df[flow_col] + threshold, 
                                  color='gray', alpha=0.2, label=f'Nguong {threshold:.2f}')
            
            axs[0].set_title(f'Du doan Flow cho cam bien {sensor_id}')
            axs[0].set_ylabel('Flow')
            axs[0].legend()
        
        # Ve ap suat 1 (Pressure_1)
        if pressure1_col in df.columns:
            axs[1].plot(df['Timestamp'], df[pressure1_col], 'g-', label='Pressure 1')
            
            # Ve diem du doan
            if 'Pressure_1' in prediction:
                axs[1].scatter([next_timestamp], [prediction['Pressure_1']], color='red', s=100, label='Du doan')
            
            axs[1].set_title(f'Du lieu Pressure 1 cho cam bien {sensor_id}')
            axs[1].set_ylabel('Pressure 1')
            axs[1].legend()
        
        # Ve ap suat 2 (Pressure_2) hoac chenh lech ap suat neu ca hai ton tai
        if pressure1_col in df.columns and pressure2_col in df.columns:
            axs[2].plot(df['Timestamp'], df[pressure2_col], 'r-', label='Pressure 2')
            
            # Ve chenh lech ap suat
            df['Pressure_Diff'] = df[pressure1_col] - df[pressure2_col]
            axs[2].plot(df['Timestamp'], df['Pressure_Diff'], 'm-', label='Chenh lech ap suat')
            
            # Ve diem du doan
            if 'Pressure_2' in prediction:
                axs[2].scatter([next_timestamp], [prediction['Pressure_2']], color='red', s=100, label='Du doan P2')
            
            if 'Pressure_Diff' in prediction:
                axs[2].scatter([next_timestamp], [prediction['Pressure_Diff']], color='purple', s=100, label='Du doan diff')
            
            axs[2].set_title(f'Du lieu Pressure 2 va Chenh lech ap suat cho cam bien {sensor_id}')
            axs[2].set_ylabel('Pressure 2 / Chenh lech')
            axs[2].legend()
        elif pressure1_col in df.columns:
            axs[2].plot(df['Timestamp'], df[pressure1_col], 'g-', label='Pressure 1 (chi co 1 diem do)')
            
            # Ve diem du doan
            if 'Pressure_1' in prediction:
                axs[2].scatter([next_timestamp], [prediction['Pressure_1']], color='red', s=100, label='Du doan')
            
            axs[2].set_title(f'Du lieu Pressure cho cam bien {sensor_id} (chi co 1 diem do)')
            axs[2].set_ylabel('Pressure')
            axs[2].legend()
        
        # Ve them bieu do error sigma
        if 'error_sigma' in anomaly_df.columns:
            fig.set_size_inches(12, 20)
            plt.subplots_adjust(hspace=0.3)
            
            # Them subplot
            ax_sigma = fig.add_subplot(4, 1, 4)
            
            valid_mask = ~anomaly_df['error_sigma'].isna()
            ax_sigma.plot(anomaly_df.loc[valid_mask, 'Timestamp'], 
                          anomaly_df.loc[valid_mask, 'error_sigma'], 
                          'k-', label='Error Sigma')
            
            # Them duong nguong 3-sigma
            ax_sigma.axhline(y=3, color='r', linestyle='--', label='Nguong 3-sigma')
            
            # Danh dau diem bat thuong
            if 'anomaly' in anomaly_df.columns:
                anomaly_points = anomaly_df[anomaly_df['anomaly'] == 1]
                if not anomaly_points.empty:
                    ax_sigma.scatter(anomaly_points['Timestamp'], anomaly_points['error_sigma'], 
                                   color='orange', s=80, marker='*', label='Bat thuong')
            
            ax_sigma.set_title('Do lech so voi Standard Error')
            ax_sigma.set_ylabel('So lan SE')
            ax_sigma.set_xlabel('Thoi gian')
            ax_sigma.legend()
        
        # Cau hinh truc x
        plt.gcf().autofmt_xdate()
        
        # Tu dong dieu chinh khoang cach
        plt.tight_layout()
        
        # Luu bieu do
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Da luu bieu do tai: {output_path}")
    except Exception as e:
        logger.error(f"Loi tao bieu do: {str(e)}")


def main():
    """Ham chinh de chay du doan."""
    # Phan tich tham so dong lenh
    args = parse_args()
    
    # Tao thu muc neu chua ton tai
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    logger.info(f"Dang tai du lieu tu: {args.data_path}")
    
    # Tai va chuyen doi du lieu
    merged_df = load_data(args.data_path)
    if merged_df is None or merged_df.empty:
        logger.error("Khong the tai du lieu.")
        return 1
    
    result_df = transform_df(merged_df)
    
    # Xac dinh cac cam bien co du ba chi so (Flow, Pressure_1, Pressure_2)
    def get_sensor_ids_with_both_pressures(df):
        filtered_columns = df.columns[df.columns.str.endswith('_Pressure_2')]
        return [col.split('_')[0] for col in filtered_columns if f"{col.split('_')[0]}_Flow" in df.columns]
    
    # Xac dinh cac cam bien chi co mot diem do ap suat
    def get_sensor_ids_with_one_pressure(df):
        all_sensors = set()
        for col in df.columns:
            if "_Pressure_1" in col:
                sensor_id = col.split('_')[0]
                if f"{sensor_id}_Flow" in df.columns and f"{sensor_id}_Pressure_2" not in df.columns:
                    all_sensors.add(sensor_id)
        return list(all_sensors)
    
    # Xac dinh tat ca cac cam bien co du lieu Flow
    def get_all_flow_sensors(df):
        all_sensors = set()
        for col in df.columns:
            if "_Flow" in col:
                all_sensors.add(col.split('_')[0])
        return list(all_sensors)
    
    # Lay danh sach cac cam bien
    sensors_with_both_pressures = get_sensor_ids_with_both_pressures(result_df)
    sensors_with_one_pressure = get_sensor_ids_with_one_pressure(result_df)
    all_flow_sensors = get_all_flow_sensors(result_df)
    
    logger.info(f"Cam bien co ca 2 diem do ap suat: {sensors_with_both_pressures}")
    logger.info(f"Cam bien chi co 1 diem do ap suat: {sensors_with_one_pressure}")
    logger.info(f"Tat ca cam bien co du lieu Flow: {all_flow_sensors}")
    
    # Su dung cam bien da chi dinh hoac tat ca cam bien
    if args.sensor_id:
        if args.sensor_id in all_flow_sensors:
            target_sensors = [args.sensor_id]
        else:
            logger.error(f"Khong tim thay cam bien {args.sensor_id} hoac khong co du lieu Flow.")
            return 1
    else:
        target_sensors = all_flow_sensors
    
    logger.info(f"Dang xu ly cam bien: {target_sensors}")
    
    # Cac cot tinh nang de du doan
    feature_cols = ['Flow', 'Pressure_1', 'Pressure_2']
    
    # Tao ket qua du doan
    all_predictions = {}
    all_anomalies = {}
    
    # Xu ly tung cam bien
    for sensor_id in target_sensors:
        try:
            has_both_pressures = sensor_id in sensors_with_both_pressures
            
            # 1. Du doan gia tri tiep theo
            logger.info(f"Dang du doan gia tri tiep theo cho cam bien {sensor_id}...")
            
            if has_both_pressures:
                logger.info(f"Cam bien {sensor_id} co ca 2 diem do ap suat, su dung mo hinh nang cao")
                try:
                    # Loc du lieu
                    sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols]
                    sensor_df = result_df[['Timestamp'] + [col for col in sensor_cols if col in result_df.columns]].copy()
                    
                    # Tao bo du doan nang cao
                    predictor = EnhancedTimeSeriesPredictor(window_size=24, use_pressure_diff=True)
                    
                    # Chuyen doi timestamp
                    sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
                    ts_df = sensor_df.set_index('Timestamp')
                    
                    # Huan luyen va du doan
                    predictor.fit(ts_df)
                    pred_values = predictor.predict_next(ts_df)
                    
                    # Chuyen ket qua thanh dict
                    prediction = {}
                    for i, col in enumerate(feature_cols):
                        if i < len(pred_values) and f"{sensor_id}_{col}" in sensor_df.columns:
                            prediction[col] = float(pred_values[i])
                    
                    # Them chenh lech ap suat neu co ca 2 diem do ap suat
                    if 'Pressure_1' in prediction and 'Pressure_2' in prediction:
                        prediction['Pressure_Diff'] = prediction['Pressure_1'] - prediction['Pressure_2']
                    
                except Exception as e:
                    logger.error(f"Loi khi du doan cho cam bien {sensor_id}: {str(e)}")
                    continue
            else:
                logger.info(f"Cam bien {sensor_id} chi co 1 diem do ap suat, su dung mo hinh tuong quan")
                prediction = predict_with_one_pressure(
                    result_df, 
                    sensor_id, 
                    ['Flow', 'Pressure_1'], 
                    similar_sensors=sensors_with_both_pressures,
                    model_dir=args.model_dir
                )
            
            if prediction:
                # Luu du doan vao ket qua
                last_timestamp = result_df['Timestamp'].max()
                next_timestamp = last_timestamp + timedelta(minutes=15)
                
                all_predictions[sensor_id] = {
                    "timestamp": next_timestamp.isoformat(),
                    "features": prediction,
                    "has_both_pressures": has_both_pressures
                }
                
                logger.info(f"Cam bien {sensor_id} - Du doan cho tick tiep theo:")
                for feature, value in prediction.items():
                    logger.info(f"  {feature}: {value:.4f}")
            else:
                logger.error(f"Khong the du doan cho cam bien {sensor_id}")
                continue
            
            # 2. Phat hien bat thuong theo phuong phap standard error
            logger.info(f"Dang phat hien bat thuong cho cam bien {sensor_id} theo phuong phap Standard Error...")
            anomaly_df = detect_anomalies(result_df, sensor_id, ['Flow', 'Pressure_1', 'Pressure_2'])
            
            if anomaly_df is not None:
                # Dem so diem bat thuong
                anomaly_count = anomaly_df['anomaly'].sum()
                logger.info(f"Cam bien {sensor_id}: Phat hien {anomaly_count} diem bat thuong (trong tong so {len(anomaly_df)} diem)")
                
                # Luu bat thuong vao ket qua
                anomaly_timestamps = anomaly_df[anomaly_df['anomaly'] == 1]['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                all_anomalies[sensor_id] = {
                    "total_points": len(anomaly_df),
                    "anomaly_count": int(anomaly_count),
                    "anomaly_timestamps": anomaly_timestamps,
                    "standard_error": float(anomaly_df['threshold'].iloc[0] / 3) if not anomaly_df['threshold'].empty else None
                }
                
                # Tao truc quan hoa neu yeu cau
                if args.visualize:
                    sensor_cols = [col for col in result_df.columns if col.startswith(f"{sensor_id}_")]
                    if sensor_cols:
                        # Tao truc quan hoa du doan va bat thuong
                        sensor_df = result_df[['Timestamp'] + sensor_cols].copy()
                        plot_predictions_with_anomalies(
                            sensor_df,
                            anomaly_df,
                            prediction,
                            sensor_id,
                            next_timestamp,
                            os.path.join(args.output_dir, f"{sensor_id}_prediction_anomalies.png")
                        )
                
        except Exception as e:
            logger.error(f"Loi khi xu ly cam bien {sensor_id}: {str(e)}")
    
    # Tao bao cao tong hop
    report_data = {
        "processed_time": datetime.now().isoformat(),
        "data_path": args.data_path,
        "sensors_summary": {
            "total_flow_sensors": len(all_flow_sensors),
            "sensors_with_both_pressures": len(sensors_with_both_pressures),
            "sensors_with_one_pressure": len(sensors_with_one_pressure),
            "processed_sensors": len(target_sensors)
        },
        "anomaly_summary": {
            "total_anomalies": sum(data["anomaly_count"] for data in all_anomalies.values()),
            "sensors_with_anomalies": sum(1 for data in all_anomalies.values() if data["anomaly_count"] > 0)
        },
        "sensor_details": {
            sensor_id: {
                "has_both_pressures": sensor_id in sensors_with_both_pressures,
                "prediction": all_predictions.get(sensor_id, {}),
                "anomalies": all_anomalies.get(sensor_id, {})
            } for sensor_id in target_sensors
        }
    }
    
    # Luu du doan vao file
    prediction_file = os.path.join(args.output_dir, f"enhanced_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(prediction_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    # Luu bat thuong vao file
    anomaly_file = os.path.join(args.output_dir, f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(anomaly_file, 'w', encoding='utf-8') as f:
        json.dump(all_anomalies, f, indent=2, ensure_ascii=False)
    
    # Luu bao cao tong hop
    report_file = os.path.join(args.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Du doan da luu vao: {prediction_file}")
    logger.info(f"Bat thuong da luu vao: {anomaly_file}")
    logger.info(f"Bao cao tong hop da luu vao: {report_file}")
    
    return 0


if __name__ == "__main__":
    try:
        print("Bat dau doan_forecasting.py")
        print("Python version:", sys.version)
        print("Current directory:", os.getcwd())
        print("Arguments:", sys.argv)
        exit_code = main()
        print("Ket thuc voi ma thoat:", exit_code)
        exit(exit_code)
    except Exception as e:
        print(f"Loi khong xu ly duoc: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1) 