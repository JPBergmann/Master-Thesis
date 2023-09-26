import subprocess

# Template for the optimization script
optimization_script_template = """
# Auto-generated script for optimizing <TICKER> at index <INDEX>
import sys
sys.path.append(".")  # Add the directory containing the template script
from hp_opt_it_prices_monthly_1mo import tune

# Replace with the actual ticker and index
tune("<TICKER>", <INDEX>)
"""

with open("./DATA/Tickers/month_tickers_clean.txt", "r") as f:
    tickers = f.read().strip().split("\n")

failed = [353, 169, 280, 221, 210, 235, 203, 400, 205, 180, 16, 129, 118, 347, 392, 277, 225, 265, 237, 25]
failed.sort()
# Loop through tickers
for idx in failed:
    ticker = tickers[idx]
    print()
    print(f"Tuning {ticker}...")
    print()
    # Create a unique script for each ticker
    optimization_script = optimization_script_template.replace("<TICKER>", ticker).replace("<INDEX>", str(idx))
    
    # Write the script to a temporary file
    script_filename = f'optimize_{idx}.py'
    with open(script_filename, 'w') as script_file:
        script_file.write(optimization_script)
    
    # Run the script using subprocess
    try:
        subprocess.run(['python', script_filename], check=True)
        # Remove the temporary script file
        subprocess.run(['rm', script_filename], check=True)
    except:
        # Remove the temporary script file
        subprocess.run(['rm', script_filename], check=True)
        # Handle errors if needed
        print(f"Failed to tune {ticker}...")
        continue
