import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_assets():
    logger.info("Loading data...")
    df = pd.read_csv('datasets/transactions.csv', skiprows=1, sep=';', quotechar="'", encoding='cp1251')
    df['transdatetime'] = pd.to_datetime(df['transdatetime'])
    
    # 1. Target Distribution
    logger.info("Generating target distribution plot...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='target', data=df, palette='viridis')
    plt.title('Class Distribution (0: Legit, 1: Fraud)')
    plt.xlabel('Target')
    plt.ylabel('Count')
    plt.savefig('plot_target_dist.png')
    plt.close()
    
    # 2. Transactions over time
    logger.info("Generating time series plot...")
    df['date'] = df['transdatetime'].dt.date
    daily_counts = df.groupby('date')['target'].count()
    daily_fraud = df[df['target'] == 1].groupby('date')['target'].count()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_counts.index, daily_counts.values, label='Total Transactions', color='blue', alpha=0.5)
    plt.plot(daily_fraud.index, daily_fraud.values, label='Fraud Transactions', color='red')
    plt.title('Daily Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plot_time_series.png')
    plt.close()
    
    # 3. Amount Distribution (Log scale)
    logger.info("Generating amount distribution plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='amount', hue='target', bins=50, log_scale=True, palette='viridis', element='step')
    plt.title('Transaction Amount Distribution (Log Scale)')
    plt.savefig('plot_amount_dist.png')
    plt.close()

if __name__ == "__main__":
    generate_assets()
