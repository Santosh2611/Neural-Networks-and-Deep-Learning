import pandas as pd

# Task 1
with open('automobile.csv', 'r') as f:
    auto_df = pd.read_csv(f)
print("First five rows of the automobile dataset:")
print(auto_df.head())
print("\nLast five rows of the automobile dataset:")
print(auto_df.tail())

# Task 2
auto_df.replace('?', pd.NA, inplace=True) # Replace the '?' symbol with NaN
auto_df.dropna(inplace=True) # Drop the rows with missing values
auto_df.to_csv('automobile_cleaned.csv', index=False) # Update the CSV file

# Task 3
# Find the most expensive car company
most_expensive_company = auto_df.loc[auto_df.price.idxmax(), 'company']
print(f"\nThe most expensive car company is {most_expensive_company}.")

# Find the least expensive car company
least_expensive_company = auto_df.loc[auto_df.price.idxmin(), 'company']
print(f"\nThe least expensive car company is {least_expensive_company}.")

# Task 4
# Print all BMW cars details
bmw_cars = auto_df[auto_df.company == 'bmw']
print("\nDetails of all BMW cars:")
print(bmw_cars)

# Task 5
# Count total cars per body-style
body_style_counts = auto_df['body-style'].value_counts()
print("\nTotal cars per body-style:")
print(body_style_counts)

# Task 6
# Find each company's highest and lowest price cars
grouped = auto_df.groupby('company')['price']
result = pd.DataFrame({
    'Highest price': grouped.max(),
    'Lowest price': grouped.min()
})
print("\nEach company's highest and lowest price cars:")
print(result)

# Task 7
# Find the average mileage of each car making company
company_mileage = auto_df.groupby('company')['average-mileage'].mean()
print("\nAverage mileage of each car making company:")
print(company_mileage)

# Task 8
# Sort all cars by price horsepower
sorted_cars = auto_df.sort_values(['price', 'horsepower'], ascending=[True, False])
print("\nAll cars sorted by price horsepower:")
print(sorted_cars)

# Task 9
# Select the data in rows [13, 24, 58] and in columns ['horsepower', 'body-style’, 'price']
selected_rows = auto_df.loc[[13, 24, 58], ['horsepower', 'body-style', 'price']]
print("\nSelected data rows:")
print(selected_rows)

# Task 10
# Select only the rows where the wheel-base is between 85 and 90, both inclusive
selected_rows = auto_df.query('85 <= `wheel-base` <= 90')
print("\nRows with wheel-base between 85 and 90:")
print(selected_rows)

# Task 11
# Change the price in row 59 to 13240
auto_df.at[59, 'price'] = 13240
print("\nPrice in row 59 updated to 13240.")
print(auto_df.loc[59])

# Task 12
# Calculate the sum of all prices
prices_sum = auto_df['price'].sum()
print("\nSum of all prices:", prices_sum)

# Task 13
# Append a new row 30 to the data frame and then delete that row to return the original DataFrame
new_row = pd.Series(['volvo', 'wagon', 15000, 28, 22, 'dohc', 4, 'mpfi', 114, 3.78, 3.15, 9.5, 121])
auto_df = auto_df.append(new_row, ignore_index=True)

# Print the updated DataFrame
print("\nDataFrame with the new row:")
print(auto_df.tail())

# Delete the new row to return the original DataFrame
auto_df.drop(index=len(auto_df)-1, inplace=True)
print("\nDataFrame after deleting the new row:")
print(auto_df.tail())
