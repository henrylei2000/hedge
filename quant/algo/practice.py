import sys
from collections import deque

import sys
import pandas as pd

def rolling_window_average():
    # Read the initial line from stdin
    line = sys.stdin.readline()
    q, t = map(int, line.strip().split(';')[0].split(','))

    # Initialize lists for prices and units
    prices = [0.00] * t
    units = [0] * t

    # Process each trade from stdin
    for trade in line.strip().split(';')[1:]:
        price, unit = map(float, trade.split(','))

        # Update the rolling window
        prices.pop(0)
        prices.append(price)
        units.pop(0)

        # Handle units less than q
        if unit < q:
            units.append(unit)
            units[0] = min(units[0], q - unit)

        # Calculate the total and total_unit efficiently
        total = sum(p * u for p, u in zip(prices, units))
        total_unit = sum(units)

        # Print the rolling window average
        if total_unit > 0:
            average = total / total_unit
            print('%.2f' % average)
        else:
            print('0.00')


if __name__ == "__xmain__":
    rolling_window_average()


def pnl():
    inputs = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            break
        a, b, c, pnl = map(float, line.split(','))
        inputs.append([a, b, c, pnl])
    df = pd.DataFrame(inputs, columns=['Apple', 'Banana', 'Carrot', 'PnL'])
    corrs = [['A', df.Apple.corr(df.PnL)], ['B', df.Banana.corr(df.PnL)], ['C', df.Carrot.corr(df.PnL)]]
    corrs.sort(key=lambda x: x[1], reverse=True)
    ret = [x[0] for x in corrs]
    print(''.join(ret))


if __name__ == "__main__":
    pnl()


def weighted_ma(data, weights, period):
    wma = []
    if len(weights) != period:
        return wma

    for i in range(len(data) - period):
        total = 0
        for j in range(i, i + period):
            total += data[j] * weights[j - i]
        wma.append(total / period)

    return wma


data = [1, 2, 3, 4, 6, 7, 7]
weights = [1, 2]
period = 2

m = weighted_ma(data, weights, period)
# print(m)


def weighted_moving_average():
    import sys

    # Read input lines one by one
    wma = {}
    trades = []
    seq = -1
    for line in sys.stdin:
        # Remove leading and trailing whitespace (e.g., newline characters)
        line = line.strip()
        if line == '':
            break
        else:
            # Process the line (e.g., print it)
            # print("You entered:", line)
            trades += line.split(';')

    print(trades)

    for t in trades:
        trade = t.split(',')
        if int(trade[3]) < seq:
            continue
        else:
            seq = int(trade[3])

        key = trade[0]
        price = int(trade[1])
        qty = int(trade[2])
        if key not in wma:
            wma[key] = [0, 0]
        wma[key][0] = (wma[key][0] * wma[key][1] + price * qty) / (wma[key][1] + qty)
        wma[key][1] += qty
        print('%s: %.2f ' % (key, wma[key][0]), end="")

#weighted_moving_average()

"""
TheoUpdate 1 100
Trade Alice 1 1 95
Trade Bob 1 1 94
Trade Alice 1 -1 107


"""
def score():
    import sys

    # Read input lines one by one
    theo_prices = {}
    trader_scores = {}
    for line in sys.stdin:
        # Remove leading and trailing whitespace (e.g., newline characters)
        line = line.strip()
        # Process the line (e.g., print it)
        # print("You entered:", line)

        data = line.split(' ')
        if data[0] == 'TheoUpdate':
            trader_scores = {}
            theo_prices[data[1]] = int(data[2])
        elif data[0] == 'Trade':
            if data[2] in theo_prices:
                if (data[3].startswith('-') and theo_prices[data[2]] < int(data[4])) or \
                        (not data[3].startswith('-') and theo_prices[data[2]] > int(data[4])):
                    sign = 1
                else:
                    sign = -1
                if data[1] in trader_scores:
                    trader_scores[data[1]] += sign * abs(int(data[3])) * ((theo_prices[data[2]] - int(data[4])) ** 2)
                else:
                    trader_scores[data[1]] = sign * abs(int(data[3])) * ((theo_prices[data[2]] - int(data[4])) ** 2)
                print('%s %.2f' % (data[1], trader_scores[data[1]]))
        else:
            pass

        # Check if the line is empty (end of input)
        if not line:
            break

# score()