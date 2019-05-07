# Data profiling tool

Pandas/Matplotlib based tool to get basic statistical reports about data provided as CSV files.

Allows batching of jobs via json files defined as:
```json
[
  {
    "name": "example1",
    "data": [
      "worldpay/data/2015/Accidents.csv"
    ],
    "column": [
      "Date",
      "Number_of_Casualties"
    ],
    "start_time": "2015-01-01",
    "end_time": "2015-06-01",
    "resolution": "day"
  },
]
```
JSON definition:
 - name - used to name output files.
 - data - defines location of input data (can be a list of more than one source)
 - column - specifies the index and column on the the analysis is to be performed
 - start_time - ignore data prior to start_time
 - end_time - ignore data afer end_time
 - resoultion - time period over which to group the column data


Example usage:

python3 worldpay.py --name example1 --data data/2015/Accidents.csv --column Date --column Number_of_Casualties --resolution day

![alt text](https://github.com/AliJeewajee/worldpay/blob/master/output/example.png)
