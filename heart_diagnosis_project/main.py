import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File path to the heart disease data
file_path = './Heart_Disease_data.csv'

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print("Columns:", data.columns)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean the data by handling missing values and ensuring correct data types.

    Parameters:
        df (DataFrame): The data to clean.

    Returns:
        DataFrame: Cleaned data.
    """
    try:
        # Dropping duplicates
        df.drop_duplicates(inplace=True)

        # Handling missing values
        df.fillna(method='ffill', inplace=True)

        # Convert relevant columns to appropriate data types if necessary
        # Example: Converting a column to datetime (if applicable)
        # df['Date'] = pd.to_datetime(df['Date'])

        print("Data cleaned successfully.")
        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return None

def perform_eda(df):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.

    Parameters:
        df (DataFrame): The data to analyze.
    """
    try:
        # Descriptive statistics
        print("Descriptive Statistics:")
        print(df.describe())

        # Distribution of target variable
        plt.figure(figsize=(8, 6))
        sns.countplot(x='target', data=df)
        plt.title('Distribution of Heart Disease')
        plt.xlabel('Heart Disease (1 = Yes, 0 = No)')
        plt.ylabel('Count')
        plt.show()

        # Correlation matrix
        correlation_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

        print("EDA completed successfully.")
    except Exception as e:
        print(f"Error in EDA: {e}")

def visualize_key_metrics(df):
    """
    Visualize key metrics and relationships in the data.

    Parameters:
        df (DataFrame): The data to visualize.
    """
    try:
        # Visualize distribution of age
        plt.figure(figsize=(8, 6))
        sns.histplot(df['age'], bins=20, kde=True)
        plt.title('Distribution of Age')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()

        # Visualize heart disease by gender
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sex', hue='target', data=df)
        plt.title('Heart Disease by Gender')
        plt.xlabel('Gender (1 = Male, 0 = Female)')
        plt.ylabel('Count')
        plt.show()

        # Visualize heart disease by chest pain type
        plt.figure(figsize=(8, 6))
        sns.countplot(x='cp', hue='target', data=df)
        plt.title('Heart Disease by Chest Pain Type')
        plt.xlabel('Chest Pain Type')
        plt.ylabel('Count')
        plt.show()

        print("Key metrics visualization completed.")
    except Exception as e:
        print(f"Error in visualizing key metrics: {e}")

def main():
    """
    Main function to execute the analysis workflow.
    """
    data = load_data(file_path)
    if data is not None:
        cleaned_data = clean_data(data)
        if cleaned_data is not None:
            perform_eda(cleaned_data)
            visualize_key_metrics(cleaned_data)
            # Save cleaned data for further use
            cleaned_data.to_csv('cleaned_heart_disease_data.csv', index=False)
            print("Cleaned data saved successfully.")
        else:
            print("Data cleaning failed.")
    else:
        print("Data loading failed.")

if __name__ == "__main__":
    main()
