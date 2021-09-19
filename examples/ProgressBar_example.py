"""
Author : Abdullah Al Masud\n
email : abdullahalmasud.buet@gmail.com\n
LICENSE : MIT License
"""

from msdlib.msd import ProgressBar
import time


# defining an array as an iterator for running a loop
arr = [i * 5 - 1 for i in range(20)]


# Example 1
# we want to calculate the value
out = []
looper = range(len(arr))    # definition of iterator
with ProgressBar(looper, 'test-array') as pbar:
    for i in looper:
        out.append(arr[i] ** 5 + 2)
        # to see the progress bar properly, I am slowing down the loop caculation using time.sleep()
        time.sleep(1.5)  # obviously you should not use this inside your loop
        pbar.inc()
print(out)


# Example 2
out = []
looper = range(1000000)    # definition of iterator
with ProgressBar(looper, 'test-array') as pbar:
    for i in looper:
        out.append(i * 5 + 2)
        pbar.inc()
print(len(out))
