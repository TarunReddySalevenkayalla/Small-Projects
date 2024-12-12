import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import matplotlib.dates as mdates

# Load the dataset
df = pd.read_csv('airline3.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract year, month, and day for further analysis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# (B) Fourier Transform of the daily passenger number variation
# Time series of passenger numbers (convert to NumPy array)
daily_passengers = np.array(df['Number'])

# Fourier Transform
N = len(daily_passengers)
T = 1  # daily data
x = np.arange(N)
yf = fft(daily_passengers)
xf = fftfreq(N, T)[:N//2]

# (C) Distribution of average daily number of passengers by month
avg_daily_passengers_monthly = df.groupby('Month')['Number'].mean()

# (D) Fourier Series approximation (first 8 terms)
n_terms = 8
fourier_series = np.zeros(N)

for n in range(n_terms):
    # Fourier coefficients (complex exponential terms)
    a_n = np.real(yf[n])
    b_n = np.imag(yf[n])
    
    # Fourier series reconstruction (only real part for approximation)
    fourier_series += a_n * np.cos(2 * np.pi * n * x / N) + b_n * np.sin(2 * np.pi * n * x / N)

# (E) Power Spectrum (contribution of different periods)
power_spectrum = 2.0 / N * np.abs(yf[:N//2])

# (F) Calculation of X and Y (values depending on the last digit of your student ID)
# Last digit of student ID is '1'
# X: Average ticket price
average_ticket_price = df['Price'].mean()

# Y: The period corresponding to the highest power in the power spectrum
# Find the index of the maximum power
max_power_index = np.argmax(power_spectrum)
# The corresponding frequency (in cycles per day)
main_period_frequency = xf[max_power_index]
# Convert frequency to period (1/frequency)
main_period_length = 1 / main_period_frequency

X = average_ticket_price  # Average ticket price
Y = main_period_length  # Main period (highest power)

# Plotting Figure 1: Monthly average number of passengers & Fourier series
plt.figure(figsize=(12, 6))

# Plotting the monthly distribution
plt.subplot(1, 2, 1)
plt.bar(avg_daily_passengers_monthly.index, avg_daily_passengers_monthly.values, color='skyblue', label='Monthly Avg. Passengers')
plt.xlabel('Month')
plt.ylabel('Average Passengers (Thousands)')
plt.title('Average Daily Passengers per Month')
plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()

# Plotting the Fourier series
plt.subplot(1, 2, 2)
plt.plot(x, daily_passengers, label='Daily Passengers', color='black', linestyle='-', alpha=0.5)
plt.plot(x, fourier_series, label=f'Fourier Series (1-8 terms)', color='orange')
plt.xlabel('Days')
plt.ylabel('Number of Passengers (Thousands)')
plt.title('Fourier Series Approximation of Daily Passengers')
plt.legend()

# Add student ID to the figure
plt.figtext(0.5, 0.01, f'Student ID: 23098031', ha='center', fontsize=10)

# Show the plots
plt.tight_layout()
plt.show()

# Plotting Figure 2: Power Spectrum
plt.figure(figsize=(10, 6))
plt.bar(xf[:N//2], power_spectrum, width=0.1, color='purple', label='Power Spectrum')
plt.xlabel('Frequency (1/Days)')
plt.ylabel('Power')
plt.title('Power Spectrum of Daily Passenger Number Variation')
plt.legend()

# Add student ID to the figure
plt.figtext(0.5, 0.01, f'Student ID: 23098031', ha='center', fontsize=10)

# Display the calculated values X and Y
plt.figtext(0.5, 0.92, f'Average Ticket Price (X): {X:.2f} Euros', ha='center', fontsize=10)
plt.figtext(0.5, 0.88, f'Main Period Length (Y): {Y:.2f} Days', ha='center', fontsize=10)

# Show the power spectrum plot
plt.tight_layout()
plt.show()

# Print X and Y values
print(f'Average Ticket Price (X): {X:.2f} Euros')
print(f'Main Period Length (Y): {Y:.2f} Days')
