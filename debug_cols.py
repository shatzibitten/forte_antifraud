import pandas as pd

with open('datasets/transactions.csv', 'r', encoding='cp1251') as f:
    for i in range(5):
        print(f"{i}: {f.readline().strip()}")

