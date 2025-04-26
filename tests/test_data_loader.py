import os
import pandas as pd
import pytest

from water_leakage.data.data_loader import DataLoader, load_data

def create_csv(tmp_path, filename, data):
    path = tmp_path / filename
    pd.DataFrame(data).to_csv(path, index=False)
    return str(path)

@pytest.fixture
def data_loader(tmp_path):
    # Khởi tạo thư mục tạm với các file CSV mẫu
    data1 = {'a': [1, 2]}
    data2 = {'b': [3, 4]}
    data3 = {'c': [5, 6]}
    ch_data = {'type': ['x', 'y']}
    create_csv(tmp_path, 'find_query_1.csv', data1)
    create_csv(tmp_path, 'find_query_2.csv', data2)
    create_csv(tmp_path, 'find_query_3.csv', data3)
    create_csv(tmp_path, 'channel_data_type.csv', ch_data)
    return DataLoader(str(tmp_path))


def test_load_all_data(data_loader):
    data = data_loader.load_all_data()
    assert 'query_1' in data and 'query_2' in data and 'query_3' in data and 'merged_data' in data
    # Kiểm tra merged_data có độ dài đúng
    assert len(data['merged_data']) == len(data['query_1']) + len(data['query_2']) + len(data['query_3'])


def test_get_sensor_ids_empty():
    df_empty = pd.DataFrame()
    dl = DataLoader('')
    assert dl.get_sensor_ids(df_empty) == []


def test_get_sensor_ids():
    df = pd.DataFrame({'A_Pressure_2': [1, 2], 'B_Pressure_2': [3, 4], 'C': [5, 6]})
    dl = DataLoader('')
    ids = dl.get_sensor_ids(df)
    assert set(ids) == {'A', 'B'}


def test_filter_sensors_with_complete_data():
    df = pd.DataFrame({
        'Timestamp': [1, 2],
        'A_Flow': [0.1, 0.2],
        'A_Pressure_1': [1, 2],
        'A_Pressure_2': [2, 3],
        'B_Pressure_1': [1, 2],
    })
    dl = DataLoader('')
    filtered = dl.filter_sensors_with_complete_data(df)
    # Chỉ sensor A có đủ dữ liệu
    assert list(filtered.columns) == ['Timestamp', 'A_Flow', 'A_Pressure_1', 'A_Pressure_2']


def test_get_data_file_info(tmp_path):
    # Không có file nào tồn tại
    dl = DataLoader(str(tmp_path))
    info = dl.get_data_file_info()
    for v in info.values():
        assert v['exists'] is False
        assert v['size'] == 0
    # Tạo một file mẫu
    path1 = tmp_path / 'find_query_1.csv'
    path1.write_text('a,b\n1,2')
    info2 = dl.get_data_file_info()
    assert info2['query_1']['exists'] is True
    assert info2['query_1']['size'] > 0


def test_load_data(tmp_path):
    # Kiểm tra hàm load_data trả về merged_data
    data = {'x': [1]}
    pd.DataFrame(data).to_csv(tmp_path / 'find_query_1.csv', index=False)
    pd.DataFrame(data).to_csv(tmp_path / 'find_query_2.csv', index=False)
    pd.DataFrame(data).to_csv(tmp_path / 'find_query_3.csv', index=False)
    pd.DataFrame(data).to_csv(tmp_path / 'channel_data_type.csv', index=False)
    df = load_data(str(tmp_path))
    assert not df.empty
    assert isinstance(df, pd.DataFrame) 