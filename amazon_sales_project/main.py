import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = './Amazon-Sales-data.csv'

# Function to load data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("Columns:", data.columns)  # Print column names to verify
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to clean data
def clean_data(df):
    try:
        # Dropping duplicates
        df.drop_duplicates(inplace=True)
        
        # Handling missing values
        df.ffill(inplace=True)  # Forward fill missing values
        
        # Ensure 'Order Date' column exists before attempting conversion
        if 'Order Date' in df.columns:
            # Converting 'Order Date' column to datetime
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            
            # Extracting year, month and year-month from 'Order Date'
            df['Year'] = df['Order Date'].dt.year
            df['Month'] = df['Order Date'].dt.month
            df['YearMonth'] = df['Order Date'].dt.to_period('M')
        else:
            raise ValueError("Column 'Order Date' not found in the dataset.")

        # Ensure 'Total Revenue' is numeric
        df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce')
        df.dropna(subset=['Total Revenue'], inplace=True)

        print("Data cleaned successfully.")
        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

# Function to analyze sales trends
def analyze_sales_trends(df):
    try:
        # Month-wise sales trend
        monthly_sales = df.groupby('Month')['Total Revenue'].sum()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=monthly_sales)
        plt.title('Month-wise Sales Trend')
        plt.xlabel('Month')
        plt.ylabel('Total Revenue')
        plt.grid(True)
        plt.show()

        # Year-wise sales trend
        yearly_sales = df.groupby('Year')['Total Revenue'].sum()
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=yearly_sales)
        plt.title('Year-wise Sales Trend')
        plt.xlabel('Year')
        plt.ylabel('Total Revenue')
        plt.grid(True)
        plt.show()

        # Yearly month-wise sales trend
        df['YearMonthStr'] = df['YearMonth'].astype(str)
        yearly_month_sales = df.groupby('YearMonthStr')['Total Revenue'].sum()
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=yearly_month_sales)
        plt.title('Yearly Month-wise Sales Trend')
        plt.xlabel('Year-Month')
        plt.ylabel('Total Revenue')
        plt.grid(True)
        plt.show()

        print("Sales trend analysis completed.")
    except Exception as e:
        print(f"Error in sales trend analysis: {e}")

# Function to analyze key metrics and correlations
def analyze_key_metrics(df):
    try:
        # Ensure 'Total Revenue' is numeric
        if df['Total Revenue'].dtype != 'float64' and df['Total Revenue'].dtype != 'int64':
            df['Total Revenue'] = pd.to_numeric(df['Total Revenue'], errors='coerce')

        # Key metrics
        total_sales = df['Total Revenue'].sum()
        average_order_value = df['Total Revenue'].mean()
        total_orders = df.shape[0]

        print(f"Total Sales: {total_sales}")
        print(f"Average Order Value: {average_order_value}")
        print(f"Total Orders: {total_orders}")

        # Correlation analysis
        df_numeric = df.select_dtypes(include=[np.number])
        correlation_matrix = df_numeric.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

        print("Key metrics and correlation analysis completed.")
    except Exception as e:
        print(f"Error in key metrics analysis: {e}")

# Main function to execute the analysis
def main():
    data = load_data(file_path)
    if data is not None:
        cleaned_data = clean_data(data)
        if cleaned_data is not None:
            analyze_sales_trends(cleaned_data)
            analyze_key_metrics(cleaned_data)
            # Save cleaned data for further use
            cleaned_data.to_csv('cleaned_amazon_sales_data.csv', index=False)
            print("Cleaned data saved successfully.")
        else:
            print("Data cleaning failed.")
    else:
        print("Data loading failed.")

if __name__ == "__main__":
    main()
