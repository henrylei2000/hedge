import numpy as np

import pandas as pd

dates = pd.date_range("20130101", periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))


print(df.mean(1))


s = pd.Series(np.random.randint(0, 7, size=10))

print(s)

vc = s.value_counts()
print(vc)

