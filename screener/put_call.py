import yfinance as yf
import matplotlib.pyplot as plt
import datetime

def calculate_put_call_ratio(stock_symbol, start_date, end_date):
    try:
        # Download options data for the stock within the specified time frame
        options_data = yf.Ticker(stock_symbol).option_chain(start=start_date, end=end_date)

        # Extract the data for calls and puts
        calls = options_data.calls
        puts = options_data.puts

        # Calculate the total volume of calls and puts
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()

        # Calculate the put/call ratio
        put_call_ratio = total_put_volume / total_call_volume

        return put_call_ratio

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    # Input the stock symbol
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ")

    # Ask the user if they want to draw the Put/Call Ratio graph
    draw_graph_option = input("Do you want to draw the Put/Call Ratio graph? (y/n): ")

    if draw_graph_option.lower() == 'y':
        # Input the start date for the analysis
        start_date = input("Enter the start date (YYYY-MM-DD): ")

        # Set the end date to today
        end_date = datetime.date.today().strftime('%Y-%m-%d')

        # Calculate the put/call ratio within the specified time frame
        put_call_ratio = calculate_put_call_ratio(stock_symbol, start_date, end_date)

        if put_call_ratio is not None:
            # Print the latest put/call ratio reading
            print(f"Latest Put/Call Ratio for {stock_symbol} (from {start_date} to {end_date}): {put_call_ratio:.4f}")

            try:
                # Create a bar graph to visualize the Put/Call Ratio
                plt.bar(['Put/Call Ratio'], [put_call_ratio], color=['blue'])
                plt.title(f'Put/Call Ratio for {stock_symbol} (from {start_date} to {end_date})')
                plt.ylabel('Ratio')
                plt.show()
            except Exception as e:
                print(f"An error occurred while plotting the graph: {str(e)}")
    else:
        # Calculate the put/call ratio without drawing the graph
        put_call_ratio = calculate_put_call_ratio(stock_symbol, None, None)

        if put_call_ratio is not None:
            # Print the latest put/call ratio reading
            print(f"Latest Put/Call Ratio for {stock_symbol}: {put_call_ratio:.4f}")

if __name__ == '__main__':
    main()
