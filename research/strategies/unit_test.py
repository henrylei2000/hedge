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
