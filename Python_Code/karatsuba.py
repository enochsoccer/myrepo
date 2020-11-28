"""
Course Info:
Stanford Algorithms Specialization on Coursera
Course 1: Divide and Conquer, Sorting and Searching, and Randomized Algorithms

Title:
Week 1, Assignment #1 - Karatsuba Multiplication Algorithm

Description:
This is a recursive algorithm for multiplying two numbers.

"""

from math import log10

def digits(n):
    "Computes number of digits."
    return 1 if n == 0 else int(log10(n)) + 1

def append_zeroes(x, n):
    """Prefixes zeroes so that the number x has an even number of digits.
    x - string data type
    n - even number of digits to have
    """
    x = '0' * (n - len(x)) + x
    return x

def karatsuba(x, y):
    "Applies the Karatsuba multiplication algorithm to multiply x and y."
    n = digits(x) if digits(x) > digits(y) else digits(y)
    if n%2:  # If n is odd, then make it even
        n += 1

    """if digits(x) % 2 or digits(y) % 2:  # If either x or y has odd-number digits
        # Set n to a power of 2
        n = (digits(x) + 1) if (digits(x) > digits(y)) else (digits(y) + 1)"""

    # Prefixing with zeroes
    x_str = append_zeroes(str(x), n)
    y_str = append_zeroes(str(y), n)

    # Split into halves
    a = int(x_str[0:int(n/2)])
    b = int(x_str[int(n/2):])
    c = int(y_str[0:int(n/2)])
    d = int(y_str[int(n/2):])

    if n == 2:
        # Multiplication algorithm
        ac = a * c
        bd = b * d
        foil = (a + b) * (c + d)
    elif n > 2:  # Recursion
        ac = karatsuba(a, c)
        bd = karatsuba(b, d)
        foil = karatsuba(a + b, c + d)

    ad_plus_bc = int(foil - ac - bd)  # int() operator to avoid doubles
    result = (10**n)*ac + ((10 ** (n//2)) * ad_plus_bc) + bd  # // operator does floor division to give integer
    return int(result)


#x = input("Enter x: ")
#y = input("Enter y: ")
x = 3141592653589793238462643383279502884197169399375105820974944592
y = 2718281828459045235360287471352662497757247093699959574966967627
result = karatsuba(int(x), int(y))
print(x, "*", y, "=", result)
