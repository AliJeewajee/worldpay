import argparse
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import *
from matplotlib.ticker import PercentFormatter
import webbrowser
from pathlib import Path
import seaborn as sns
sns.set_style('whitegrid')


class Task(object):
    """
    Task class is used to define the parameters
    of how a set of data should be analysed
    """
    def __init__(self, name: str, data: List[str], column: List[str],
                 start_time: datetime.datetime, end_time: datetime.datetime, resolution: str):
        """
        :param data: list of data sources to use in task
        :param column: list of columns required by task
        :param start_time: start time/date over which analysis is to be performed
        :param end_time: end time/date over which analysis is to be performed
        :param resolution: analyse data grouped by ('weekday', 'month')
        """
        self._name = name
        self._data = data
        self._column = column
        self._start_time = start_time
        self._end_time = end_time
        self._resolution = resolution

    def __repr__(self):
        return json.dumps(
            dict(name=self._name,
                 data=self.data,
                 column=self.column,
                 start_time=self.start_time.strftime('%Y-%m-%d'),
                 end_time=self.end_time.strftime('%Y-%m-%d'),
                 resolution=self.resolution), indent=4)

    @classmethod
    def create_task_from_dict(cls, task_dict: dict):
        """
        Constructor for producing a Task object from a dictionary
        :param task_dict:
        :return:
        """
        # checks for correct form of task_dict otherwise raise exception
        task = task_dict.copy()
        start = task.pop('start_time')
        end = task.pop('end_time')

        task['start_time'] = datetime.datetime.strptime(start, '%Y-%m-%d') \
            if start else None
        task['end_time'] = datetime.datetime.strptime(end, '%Y-%m-%d') \
            if end else None
        return cls(**task)

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @property
    def column(self):
        return self._column

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def resolution(self):
        return self._resolution


class Analysis(object):
    """
    Analysis class, hosts the set of possible analyses
    """
    def __init__(self, task: Task, data: pd.DataFrame, date_format='%d/%m/%Y'):
        """
        :param task: Task object defining the parameters of the analysis
        :param data: data to be analysed
        """
        self._stats = None
        self._task = task
        self._data = data.loc[:, task.column]
        self._date_format = date_format
        self._parse_dates(task)

    def _parse_dates(self, task):
        """
        Parse the index column for dates and cast to datetime format.
        :param task: Task object defining the parameters of the analysis
        :return:
        """
        if len(task.column) > 1:
            self._data[self._task.column[0]] = pd.to_datetime(
                self._data[self._task.column[0]], format=self._date_format)
            self._data.set_index([self._task.column[0]], inplace=True)
            if task.start_time and task.start_time:
                mask = np.logical_and(self._data.index >= task.start_time, self._data.index < task.end_time)
                self._data = self._data.loc[mask, :]

    def summary_statistics(self, null_val: float = -1):
        """
        Calculate and return a set of basic statistics on data
        :param null_val: pass a value that defines the case for missing or null values
        :return: dict
        """
        data = self._data.copy()
        data.loc[data[self._task.column[1]] == null_val] = np.nan
        null_count = data.isna().sum()
        non_null_count = (~data.isna()).sum()
        data.dropna(inplace=True)
        return dict(
            length=(null_count+non_null_count).values.tolist()[0],
            count=non_null_count.values.tolist()[0],
            missing=null_count.values.tolist()[0],
            min=data.min().values.tolist()[0],
            q1=data.quantile(0.25).values.tolist()[0],
            median=data.median().values.tolist()[0],
            q3=data.quantile(0.75).values.tolist()[0],
            max=data.max().values.tolist()[0],
            mean='{:.3f}'.format(data.mean().values.tolist()[0]),
            mode=data.mode().values.tolist()[0][0],
            std='{:.3f}'.format(data.std().values.tolist()[0])
        )

    def plot_pareto(self, ax: plt.Axes):
        """
        Generate a pareto plot of distribution of column data when grouped by
        time period. Eg. distribution of no. of casualties in a day/week/month
        :param ax: matplotlib axes object
        :return:
        """
        if self._task.resolution == 'day':
            data = self._data.groupby(by=[self._data.index.dayofyear, self._data.index.year]).sum()
        elif self._task.resolution == 'week':
            data = self._data.groupby(by=[self._data.index.week, self._data.index.year]).sum()
        elif self._task.resolution == 'month':
            data = self._data.groupby(by=[self._data.index.month, self._data.index.year]).sum()
        else:
            data = self._data.copy()

        xlabel = self._task.column[1]

        if self._task.resolution == 'day':
            bins = 100
        else:
            bins = 10
        freqs, np_bins = np.histogram(data[self._task.column[1]].values, bins=bins)
        ax.hist(data.values, bins=bins)
        cumpercent = np.cumsum(freqs) / np.sum(freqs) * 100
        ax.set_xlabel(xlabel)
        ax2 = ax.twinx()
        ax2.plot(np_bins[:-1] + 0.5*np.diff(np_bins[:2]), cumpercent, color="C1", marker="D", ms=7)
        ax2.yaxis.set_major_formatter(PercentFormatter())

        ax.tick_params(axis="y", colors="C0")
        ax2.tick_params(axis="y", colors="C1")
        ax.set_ylabel('frequency')
        ax2.set_ylabel('percentage')
        ax.set_title('Distribution of %s by %s' % (self._task.column[1], self._task.resolution))
        ax.axis('tight')
        ax2.grid('off')
        return ax

    def plot_table(self, ax: plt.Axes):
        """
        Plot a table of descriptive statistical info.
        :param ax: matplotlib axes object
        :return:
        """
        row_labels = ['length', 'count', 'missing', 'min',
                      'q1', 'median', 'q3', 'max', 'mean', 'mode', 'std']
        cell_text = [[str(self._stats[row])] for row in row_labels]
        col_labels = ['statistics']
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=cell_text,
                 rowLabels=row_labels,
                 colLabels=col_labels,
                 colWidths=[0.5, 0.5],
                 loc='center')

        return ax

    def plot_counts_over_time_window(self, ax: plt.Axes):
        """
        Plot the data in column bucketed by time period, E.g. no. of casualties
        in a day, week or month, between start_time and end_time.
        :param ax: matplotlib axes object
        :return:
        """
        if self._task.resolution == 'day':
            data = self._data.groupby(by=[self._data.index.dayofyear, self._data.index.year]).sum()
        elif self._task.resolution == 'week':
            data = self._data.groupby(by=[self._data.index.week, self._data.index.year]).sum()
        elif self._task.resolution == 'month':
            data = self._data.groupby(by=[self._data.index.month, self._data.index.year]).sum()
        else:
            data = self._data.copy()
        _ = ax.plot(data[self._task.column[1]].values)[0]
        ax.set_xlabel(self._task.resolution)
        ax.set_ylabel(self._task.column[1])
        ax.axis('tight')
        return ax

    def run(self, axs: List[plt.Axes]):
        """
        Entry point for executing the analysis defined in self._task
        :param axs: matplotlib axes objects in which results are presented
        :return:
        """
        self._stats = self.summary_statistics()
        self.plot_pareto(axs[0])
        self.plot_counts_over_time_window(axs[1])
        self.plot_table(axs[2])
        return self._stats


class Processor(object):
    """
    Processor class, takes a set of tasks, loads the relevant, runs analyses,
    takes care of saving and displaying data
    """
    def __init__(self, tasks: list):
        self._data = None
        self._tasks = tasks

    @staticmethod
    def load_data_from_csv(task: Task):
        """
        Loads data from csv, if more than one sources are specified,
        attempts detedt an index on which to join them and returns a
        joined dataframe.
        :param task:
        :return:
        """
        columns_so_far = set()
        data_so_far = None
        for filename in task.data:
            table = pd.read_csv(filename)
            potential_indices = list(columns_so_far.intersection(table.columns))
            if len(potential_indices) == 0 and data_so_far is not None:
                raise Exception('Data sources have no common indices on which to join.'
                                'Specify single source or multiple sources with a common'
                                'column index names')

            columns_so_far.update(table.columns)
            if data_so_far is None:
                data_so_far = table
            else:
                data_so_far = data_so_far.merge(table, how='inner', on=potential_indices)

        if not all([tc in columns_so_far for tc in task.column]):
            raise Exception('Column "{0}" is not in the joined final table'.format(task.column))

        return data_so_far

    @staticmethod
    def save_analysis(task: Task, file_format: str = 'png'):
        """
        Save the output of the analysis as PDF/PNG
        :return:
        """
        fname = Path(__file__).resolve().parent.joinpath('output', task.name + '.' + file_format)
        plt.savefig(fname, format=file_format)
        return fname

    @staticmethod
    def disp_analysis(fname: Path):
        """
        Takes analysis output saved by save_analysis and pops it up in a browser window
        :return:
        """
        browser = webbrowser.get('chrome')
        browser.open(fname.as_uri())

    def process(self, tasks: List[Task]):
        """
        Entry point for task processor, which executes analyses
        on a given list of tasks. For each task will create a figure instance,
        load the data, run the analysis, save and display results.
        :param tasks:
        :return:
        """
        stats = None
        for subtask in tasks:
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
            fig.suptitle(f'Profile for %s \n'
                         f'Start: {subtask.start_time} \n'
                         f'End: {subtask.end_time}' % ' '.join(subtask.data))
            data = self.load_data_from_csv(subtask)
            analysis = Analysis(subtask, data)
            stats = analysis.run(axs)
            fname = self.save_analysis(subtask)
            self.disp_analysis(fname)
            print('subtask complete')

        print('task complete')
        return stats


def input_parser(argv: str = None):
    """
    Parse command line inputs and return them in a format
    for the task builder to create tasks from .
    :param argv: str
    :return: dict
    """
    parser = argparse.ArgumentParser(
        description='Parse parameters for Data Profiler',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default=None)
    parser.add_argument('--data', default=None, nargs='*')
    parser.add_argument('--column', default=None, nargs='?', action='append')
    parser.add_argument('--start_time')
    parser.add_argument('--end_time')
    parser.add_argument('--resolution', choices=['day', 'week', 'month', 'year'])
    parser.add_argument('--batch')
    params = parser.parse_args(argv)

    if params.batch:
        with open(params.batch) as batch_handle:
            inputs = json.load(batch_handle)
    else:
        inputs = [vars(params)]
        inputs[0].pop('batch')
    print(json.dumps(inputs, sort_keys=True, indent=4))
    return inputs


def main(argv: str = None):
    """
    Entry point running the profiling process.
    :param argv: str specifying

    --name: str          name of analysis
    --data: List[str]    [locations to data]
    --column: List[str]  [index column, data column] to analyse
    --start_time: str    start time from which to analyse timeseries
    --end_time: str      end time from which to analyse timeseries
    --resolution: str    time period by which to group column data
    --batch: str         location to json file defining batch jobs
    :returns:


    """
    task_dicts = input_parser(argv)
    task_list = [Task.create_task_from_dict(task_dict) for task_dict in task_dicts]
    processor = Processor(task_list)
    return processor.process(task_list)


if __name__ == '__main__':
    main()
