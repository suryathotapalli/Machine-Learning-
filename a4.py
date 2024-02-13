import pandas as pd
import statistics
import matplotlib.pyplot as plt

file_path = 'Lab Session1 Data.xlsx'
df = pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

mean_price = statistics.mean(df['Price'])
variance_price = statistics.variance(df['Price'])
print(f"Mean of Price: {mean_price}")
print(f"Variance of Price: {variance_price}")

wednesday_data = df[df['Day'] == 'Wed']
sample_mean_wednesday = statistics.mean(wednesday_data['Price'])
print(f"Sample mean for Wednesdays: {sample_mean_wednesday}")
print(f"Population mean: {mean_price}")
print("Observation: Compare the sample mean for Wednesdays with the population mean.")

april_data = df[df['Month'] == 'Apr']
sample_mean_april = statistics.mean(april_data['Price'])
print(f"Sample mean for April: {sample_mean_april}")
print(f"Population mean: {mean_price}")
print("Observation: Compare the sample mean for April with the population mean.")

loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}")

profit_wednesday_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {profit_wednesday_probability}")

conditional_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Conditional probability of making profit on Wednesday: {conditional_profit_probability}")

plt.scatter(df['Day'], df['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter Plot of Chg% against Day of the Week')
plt.show()