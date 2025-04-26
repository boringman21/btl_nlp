import os
import shutil
import pandas as pd
import pytest

from water_leakage.data.data_transform import transform_df, cache_dir


def create_minimal_df():
    """
    Tạo DataFrame tối thiểu với một sensor và các cột time/value cần thiết
    """
    rows = []
    # Tạo 3 bản ghi cho các chNumber 0, 1, 2
    for ch in [0, 1, 2]:
        row = {
            'chNumber': ch,
            'smsNumber': '1'
        }
        # Thêm các cột dataValues.X.dataTime và dataValues.X.dataValue
        for i in range(96):
            row[f'dataValues.{i}.dataTime'] = pd.Timestamp('2021-01-01')
            row[f'dataValues.{i}.dataValue'] = float(i)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def test_transform_df_caching(tmp_path):
    # Xóa thư mục cache nếu đã tồn tại trước đó
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    assert not os.path.exists(cache_dir)

    df = create_minimal_df()
    # Gọi transform_df lần đầu tiên
    result1 = transform_df(df)
    # Thư mục cache phải được tạo sau lần gọi đầu tiên
    assert os.path.isdir(cache_dir)
    assert len(os.listdir(cache_dir)) > 0

    # Gọi transform_df lần thứ hai với cùng input
    result2 = transform_df(df)
    # Kết quả trả về phải giống nhau (cache được sử dụng)
    pd.testing.assert_frame_equal(result1, result2) 