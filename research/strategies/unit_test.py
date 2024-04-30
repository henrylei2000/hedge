import pandas as pd

def analyze_rsi_series(data):
    if 'rsi' not in data.columns:
        raise ValueError("Data must include 'rsi' column")

    # Initialize variables
    result = []
    rsi_values = data['rsi'].tolist()

    # First, detect all peaks and valleys
    for i in range(2, len(rsi_values) - 2):
        # Read surrounding RSI values to determine peaks or valleys
        prev_rsi_1, prev_rsi_2 = rsi_values[i - 2], rsi_values[i - 1]
        curr_rsi = rsi_values[i]
        next_rsi_1, next_rsi_2 = rsi_values[i + 1], rsi_values[i + 2]

        # Check for a peak
        if curr_rsi > max(prev_rsi_1, prev_rsi_2, next_rsi_1, next_rsi_2):
            print(f"{i} ----- {rsi_values[i]}")
            result.append((i, curr_rsi, 'peak'))
        # Check for a valley
        elif curr_rsi < min(prev_rsi_1, prev_rsi_2, next_rsi_1, next_rsi_2):
            result.append((i, curr_rsi, 'valley'))

    return result

# Example usage:
data = pd.DataFrame({'rsi': [30, 45, 70, 60, 55, 75, 65, 50, 45, 55, 65, 30, 20]})
print(analyze_rsi_series(data))


def filter_rsi_subset(macd_list, rsi_list):
    filtered_rsi = []

    for rsi_tuple in rsi_list:
        rsi_index, rsi_price, rsi_type = rsi_tuple
        for macd_tuple in macd_list:
            macd_index, _, macd_type = macd_tuple
            if macd_type == rsi_type and macd_index > rsi_index:
                filtered_rsi.append(rsi_tuple)
                break

    return filtered_rsi


# Example usage
macd_list = [(363, -0.005207801642914234, 'valley'), (366, -0.0033239127203188445, 'peak'), (371, -0.0035541983098674734, 'valley')]
rsi_list = [(359, 57.712970069071226, 'peak'), (360, 44.41980783444175, 'valley'), (363, 49.96673320026621, 'valley'), (364, 58.33865814696503, 'peak'), (370, 40.458015267174545, 'valley'), (376, 92.307692307694, 'peak'), (377, 71.31502890173546, 'valley'), (379, 76.29850746268784, 'peak')]

filtered_rsi_subset = filter_rsi_subset(macd_list, rsi_list)
print(macd_list)
print(filtered_rsi_subset)
