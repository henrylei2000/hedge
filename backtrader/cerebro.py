from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import time

# Import the backtrader platform
import backtrader as bt
import strategy
import data.feeder as feeder

"""
Please run data/feeder.py first to save online data to csv
"""

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(strategy.Consecutive)

    ticker = "TQQQ"

    # Set our desired cash start
    cerebro.broker.setcash(10000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Now let's prepare data feed
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    one_year_ago = yesterday - datetime.timedelta(days=365)

    feeder.stock_to_csv(ticker)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, "data/" + ticker + ".csv")

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=one_year_ago,
        # Do not pass values after this date
        todate=yesterday,
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    cerebro.plot()
