from finvizfinance.screener.overview import Overview
f = Overview()
try:
    f.set_filter({'Market Cap.': 'xxx'})
except ValueError as e:
    print(str(e))
