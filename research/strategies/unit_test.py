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


def filter_rsi_subset(macd, rsi):
    if len(macd) > 3 and len(rsi) > 3:
        # RSI Lifting MACD
        # strength & velocity (interval between peaks and valleys)
        macd_points, rsi_points = [], []
        print(macd[-4:])
        print(rsi[-10:])
        for macd_index, macd_value, macd_type in reversed(macd[-4:]):
            macd_points.append((macd_index, macd_value, macd_type))
            causing_rsi = []
            for rsi_index, rsi_value, rsi_type in reversed(rsi[-10:]):
                if rsi_index < macd_index or rsi_index == macd_index:
                    if rsi_type != macd_type:
                        if len(causing_rsi):  # conclude the current search
                            break
                    else:
                        causing_rsi.append((rsi_index, rsi_value, rsi_type))
            rsi_points.append(causing_rsi)
        print(f'{macd_points} -------------------------- {rsi_points}')


# Example usage
macd = [(2, 0.005656788341390495, 'valley'), (8, 0.024355482933032135, 'peak'), (11, 0.017572528247562502, 'valley'), (27, 0.10857385642029271, 'peak')]
rsi = [(2, 41.228070175438475, 'valley'), (5, 62.85973947288657, 'peak'), (8, 50.89158345221094, 'valley'), (11, 56.01851851851898, 'peak'), (14, 39.90825688073404, 'valley'), (16, 63.359591228443854, 'peak'), (18, 58.470986869971036, 'valley'), (27, 85.73446327683581, 'peak')]

filter_rsi_subset(macd, rsi)