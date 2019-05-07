# Data profiling tool

Pandas/Matplotlib based tool to get basic statistical reports about data provided as CSV files.

Example usage:

python3 worldpay.py --name example1 --data data/2015/Accidents.csv --column Date --column Number_of_Casualties --resolution day

**NOTE to make it easier to run the script and the documentation viewable immediately, these files have been *checked in*. eg. ['/data', '/output', 'docs'], etc.**

# Features
- Produces Pareto plot, timeseries plot and table of descriptive statistics.
- Loads data sources in CSV format.
- If more than one source is specified, it will attempt to detect columns on which to join them.
- Produces results for any specified data source/column combination, over a specified time period and temporal resolution.
- Allows batching of jobs via json files defined by way of example as:
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
 - name - name of output files.
 - data - defines location of input data (can be a list of more than one source)
 - column - specifies the index and column on the the analysis is to be performed
 - start_time - ignore data prior to start_time
 - end_time - ignore data afer end_time
 - resolution - time period over which to group the column data

# Example output
![alt text](https://github.com/AliJeewajee/worldpay/blob/master/output/example.png)

# Limitations
- Works only with CSV data
- Limited analytical methods
- Data size (all operations are performed in memory)

# Future developments
- Use a proper database for versatility in accessing data.
- Something like spark for data size, performance limitations
- Add concurrency for performance especially for tasks that are I/O limited
- Expand profiling capabilities, more plots, statistical methods.
- Add a proper GUI with easier and more sophisticated data manipulation, dynamic visualisation.
