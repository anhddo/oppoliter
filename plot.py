import matplotlib.pyplot as plt
import numpy as np

n = 1000
x = np.arange(1, n, n // 100)
plt.plot(x, np.sqrt(x), label='sqrt')
plt.plot(x, np.sqrt(x * np.log(x)), label='sqrt(T * log(T))')
plt.plot(x, np.log(x), label='log')
plt.legend()
plt.show()
