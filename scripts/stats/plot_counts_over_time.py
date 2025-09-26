import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
import polars as pl

class CustomSciFormatter(ticker.ScalarFormatter):
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        super().__init__(useOffset=useOffset, useMathText=useMathText)
        self._order_of_mag = order_of_mag
        
    def _set_order_of_magnitude(self):
        # Don't auto-determine order of magnitude
        # Instead use the one passed in
        self.orderOfMagnitude = self._order_of_mag
        
    def _set_format(self):
        # Set the format string to show only the coefficient
        self.format = '%1.0f'

def main():
    figsize = (3.5, 2.5)

    hour_df = pl.read_csv('./data/stats/1hour/time_counts.csv')
    day_df = pl.read_csv('./data/stats/24hour/time_counts.csv')

    hour_df = hour_df.with_columns(pl.col('createTime').cast(pl.Datetime))
    day_df = day_df.with_columns(pl.col('createTime').cast(pl.Datetime))

    hour_df = hour_df.sort('createTime')
    day_df = day_df.sort('createTime')

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(hour_df['createTime'], hour_df['count'], alpha=0.33, s=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    myFmt = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(myFmt)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=list(range(0, 60, 20))))
    fig.tight_layout()
    fig.savefig('./figs/posts_per_second.png')

    minute_df = hour_df.group_by_dynamic('createTime', every='1m').agg(pl.col('count').sum())

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(minute_df['createTime'], minute_df['count'], marker='o', linestyle='-')
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    fig.savefig('./figs/posts_per_minute.png')

    correct_factor = minute_df['count'].mean() / minute_df.filter(pl.col('createTime').dt.minute() == 42)['count'][0]

    day_df = day_df.group_by_dynamic('createTime', every='1h').agg((pl.col('count').sum() * 60 * correct_factor))


    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(day_df['createTime'], day_df['count'], marker='o', linestyle='-')
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.set_xticks(day_df['createTime'].gather_every(4).to_list())
    date_fmt = mdates.DateFormatter('%dT%H:%M')
    ax.xaxis.set_major_formatter(date_fmt)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[22, 6, 14, 22]))
    # Now determine what order of magnitude to use based on the data
    max_value = max(day_df['count'])
    order = 0
    while max_value >= 1000:
        max_value /= 1000
        order += 3

    # Apply our custom formatter with the determined order of magnitude
    formatter = CustomSciFormatter(order_of_mag=order, useOffset=True, useMathText=True)
    ax.yaxis.set_major_formatter(formatter)
    
    # fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig('./figs/posts_per_hour.png')

    hour_df = hour_df.with_columns(pl.col('createTime').dt.second().alias('createSecond'))

    # plot count per second
    second_counts = hour_df.group_by('createSecond')\
        .agg(pl.col('count').sum().alias('count'))\
        .sort('createSecond')
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(second_counts['createSecond'], second_counts['count'])
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Videos")
    fig.tight_layout()
    fig.savefig('./figs/posts_per_each_second.png')

if __name__ == '__main__':
    main()