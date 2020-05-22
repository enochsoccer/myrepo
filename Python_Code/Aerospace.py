import numpy as np

def findNeedle(haystack,needle):
    bools = (haystack == needle[0, 0])
    rows, cols = np.where(bools == True)
    Nr_hayS, Nc_hayS = haystack.shape
    Nr_need, Nc_need = needle.shape
    rows = rows[rows <= (Nr_hayS - Nr_need)]  # pre-process valid rows
    idxes = np.where(rows <= (Nr_hayS - Nr_need))
    cols = cols[idxes]  # same for the columns

    for (r, c) in zip(rows, cols):  # investigating potential needle matches
        match = (haystack[r:r+Nr_need, c:c+Nr_need] == needle)
        if np.all(match):  # if the matrix matches, then you've found the needle in the haystack
            return r, c

    return -1  # if the function reaches here, then function isn't working properly

def genHaystack():
    low = 0; high = 9
    nrows = 8; ncols = 8
    haystack = np.random.randint(low, high, size=(nrows,ncols))
    #print("haystack\n", haystack)
    return haystack

def genNeedle(haystack):
    nrows, ncols = haystack.shape
    row = np.random.randint(0, nrows - 4)
    col = np.random.randint(0, ncols - 4)
    needle = haystack[row:row + 3, col:col + 3]
    #print("needle\n", needle)
    return row, col, needle

loop = 0
count = 0
while loop < 100:
    haystack = genHaystack()
    row, col, needle = genNeedle(haystack)
    (i, j) = findNeedle(haystack, needle)
    if row == i and col == j:
        count += 1
    loop += 1
print(count)

"""
n,m = a.shape

i,j = np.where(a == 10)  # 0 = row, 1 = column
print(i,j)

from matplotlib import pyplot as plt

x = np.arange(1,11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y,'o')
plt.show()
"""
