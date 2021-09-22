# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows',12)
pd.set_option('display.max_columns',12)
df = pd.read_csv('contents/AAPL.csv', index_col='Date', parse_dates=['Date'])

# for shift func test
# shift 함수를 사용했을 때 그 결과값이 내장 객체에 저장되지 않는다.
print(df.head(3))
print("------------------------\n")
print(df.shift(1).loc['1997-01-03', 'Adj Close'])
print("------------------------\n")
print(df.head(2))

print("------------------------\n")
print(False and True)