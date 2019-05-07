import pytest
import datetime
import worldpay.worldpay as wp
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from pathlib import Path


def test_create_task_from_dict():
    test_input = {
        "name": "test",
        "data": ["Accidents.csv", "Casualties.csv"],
        "column": ["Date", "Number_of_Casualties"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"
    }
    test_output = {
        "name": "test",
        "data": ["Accidents.csv", "Casualties.csv"],
        "column": ["Date", "Number_of_Casualties"],
        "start_time": datetime.datetime.strptime("2019-01-01", '%Y-%m-%d'),
        "end_time": datetime.datetime.strptime("2019-02-01", '%Y-%m-%d'),
        "resolution": "day"
    }
    test_task = wp.Task.create_task_from_dict(test_input)

    for key in test_output:
        assert hasattr(test_task, key)

    assert test_task.name == test_output['name']
    assert test_task.data == test_output['data']
    assert test_task.column == test_output['column']
    assert test_task.start_time == test_output['start_time']
    assert test_task.end_time == test_output['end_time']
    assert test_task.resolution == test_output['resolution']


def test_summary_statistics():
    test_input = {
        "name": "test",
        "data": ["Accidents.csv", "Casualties.csv"],
        "column": ["Date", "Number_of_Casualties"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"}

    test_task = wp.Task.create_task_from_dict(test_input)
    test_data = [['01/01/2019', 1], ['02/01/2019', 2], ['03/01/2019', 2],
                 ['04/01/2019', 2], ['05/01/2019', 3], ['06/01/2019', 3],
                 ['07/01/2019', 4], ['08/01/2019', 4], ['09/01/2019', np.nan]]

    test_df = pd.DataFrame(data=test_data, columns=test_task.column)

    test_analysis = wp.Analysis(test_task, test_df)
    test_stats = test_analysis.summary_statistics()

    expected_stats = dict(min=1.0, max=4.0, mean='2.625', median=2.5, mode=2.0,
                          missing=1, length=9, count=8, std='1.061',
                          q1=2.0, q3=3.25)
    assert test_stats == expected_stats


def test_load_data_from_csv_one_file():
    test_path = Path(__file__).parent.joinpath('test_data')
    test_task = {
        "name": "test_name",
        "data": [str(test_path.joinpath("test_load_data_from_csv_1.csv"))],
        "column": ["date", "some_data"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"
    }

    csv_cols = ['date', 'some_data', 'more_data']
    csv_data = [
        ['2019-01-01', 10, 100],
        ['2019-01-02', 20, 200],
        ['2019-01-03', 30, 300]]

    expected_df = pd.DataFrame(data=csv_data, columns=csv_cols)
    task = wp.Task.create_task_from_dict(test_task)
    test_df = wp.Processor.load_data_from_csv(task)

    assert_frame_equal(test_df, expected_df)


def test_load_data_from_csv_two_files_no_common_columns():
    test_path = Path(__file__).parent.joinpath('test_data')
    test_task = {
        "name": "test_name",
        "data": [str(test_path.joinpath("test_load_data_from_csv_1.csv")),
                 str(test_path.joinpath("test_load_data_from_csv_2.csv"))],
        "column": ["date", "some_data"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"
    }

    task = wp.Task.create_task_from_dict(test_task)
    with pytest.raises(Exception):
        wp.Processor.load_data_from_csv(task)


def test_load_data_from_csv_two_files_one_common_columns():
    test_path = Path(__file__).parent.joinpath('test_data')
    test_task = {
        "name": "test_name",
        "data": [str(test_path.joinpath("test_load_data_from_csv_1.csv")),
                 str(test_path.joinpath("test_load_data_from_csv_3.csv"))],
        "column": ["date", "some_data"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"
    }

    csv_cols = ['date', 'some_data', 'more_data', 'different_data', 'other_data']
    csv_data = [
        ['2019-01-01', 10, 100, 5, 50],
        ['2019-01-02', 20, 200, 15, 150],
        ['2019-01-03', 30, 300, 25, 250]]
    expected_df = pd.DataFrame(data=csv_data, columns=csv_cols)

    task = wp.Task.create_task_from_dict(test_task)
    test_df = wp.Processor.load_data_from_csv(task)

    assert_frame_equal(test_df, expected_df)


def test_load_data_from_csv_two_files_two_common_columns():
    test_path = Path(__file__).parent.joinpath('test_data')
    test_task = {
        "name": "test_name",
        "data": [str(test_path.joinpath("test_load_data_from_csv_1.csv")),
                 str(test_path.joinpath("test_load_data_from_csv_4.csv"))],
        "column": ["date", "some_data"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"
    }

    csv_cols = ['date', 'some_data', 'more_data', 'other_data']
    csv_data = [
        ['2019-01-01', 10, 100, 50],
        ['2019-01-02', 20, 200, 150],
        ['2019-01-03', 30, 300, 250]]
    expected_df = pd.DataFrame(data=csv_data, columns=csv_cols)

    task = wp.Task.create_task_from_dict(test_task)
    test_df = wp.Processor.load_data_from_csv(task)
    assert_frame_equal(test_df, expected_df)


def test_analysis_parse_dates():
    test_path = Path(__file__).parent.joinpath('test_data')
    test_task = {
        "name": "test_name",
        "data": [str(test_path.joinpath("test_load_data_from_csv_1.csv")),
                 str(test_path.joinpath("test_load_data_from_csv_4.csv"))],
        "column": ["date", "some_data"],
        "start_time": "2019-01-01",
        "end_time": "2019-02-01",
        "resolution": "day"
    }

    csv_cols = ['some_data']
    csv_index = pd.DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03'], name='date')
    csv_data = [10, 20, 30]
    expected_df = pd.DataFrame(data=csv_data, index=csv_index, columns=csv_cols)
    task = wp.Task.create_task_from_dict(test_task)
    test_df = wp.Processor.load_data_from_csv(task)
    analysis = wp.Analysis(task, test_df, '%Y-%m-%d')
    assert_frame_equal(analysis._data, expected_df)


def test_worldpay_acceptance():
    expected_stats = dict(min=1.0, max=3.0, mean='1.107', median=1.0, mode=1.0,
                          missing=0, length=84, count=84, std='0.348',
                          q1=1.0, q3=1.0)


    test_stats = wp.main([
        '--name', 'acceptance',
        '--data', '/Users/Jeeves/Git/interviews/worldpay/worldpay/tests/test_data/test_worldpay_acceptance.csv',
        '--column', 'Date', '--column', 'Number_of_Casualties',
        '--resolution', 'day'])
    assert test_stats == expected_stats
