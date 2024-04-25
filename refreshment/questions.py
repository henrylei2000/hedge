"""
Is Unique: Implement an algorithm to determine if a string has all unique characters. What if you cannot use additional data structures?

This question asks you to write a program that takes in a string as input and determines whether all the characters in the string are unique. In other words, the program should check if there are any repeated characters in the string.

The follow-up question asks what you would do if you couldn't use additional data structures to solve the problem, which means you would have to solve it in-place with the given string.

Here's an example:

Input: "hello"

Output: False

Input: "world"

Output: True
"""


def is_unique(s):
    uni = True
    for i in range(len(s)):
        if s[i] in s[i+1:]:
            uni = False

    return uni


def is_unique_with_set(s):
    char_set = set()
    for char in s:
        if char in char_set:
            return False
        char_set.add(char)
    return True


def is_unique_with_table(s):
    freq = {}
    for c in s:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1

    for c in s:
        if freq[c] > 1:
            return False

    return True


print(is_unique_with_set('world'))


"""
1.2 Check Permutation
Given two strings, write a method to decide if one is a permutation of the other.
"""


def check_permutation(s1, s2):
    if sorted(s1) == sorted(s2):
        return True
    else:
        return False


print(check_permutation('abcd', 'dbac'))


import numpy as np
import matplotlib.pyplot as plt

def time_frac_diffusion_wave_equation(alpha, L, T, N, M):
    """
    Solves the time-fractional diffusion-wave equation using the Finite Difference Method.

    Parameters:
        alpha (float): Fractional order (0 < alpha < 1).
        L (float): Length of the domain.
        T (float): Total simulation time.
        N (int): Number of spatial grid points.
        M (int): Number of time steps.

    Returns:
        u (2D array): Solution matrix.
    """
    dx = L / N
    dt = T / M

    x = np.linspace(0, L, N)
    u = np.zeros((M+1, N))

    # Initial condition
    u[0, :] = np.sin(np.pi * x)

    for k in range(1, M+1):
        for i in range(1, N-1):
            u[k, i] = u[k-1, i] + (alpha * dt / dx**2) * (u[k-1, i+1] - 2*u[k-1, i] + u[k-1, i-1])

    return u

# Parameters
alpha = 0.5  # Fractional order (0 < alpha < 1)
L = 1.0  # Length of the domain
T = 1.0  # Total simulation time
N = 100  # Number of spatial grid points
M = 1000  # Number of time steps

# Solve the equation
solution = time_frac_diffusion_wave_equation(alpha, L, T, N, M)

# Filter out NaN or Inf values
solution_filtered = np.where(np.isnan(solution) | np.isinf(solution), 0, solution)

# Plotting
plt.figure(figsize=(8, 6))
plt.imshow(solution_filtered, extent=[0, L, 0, T], aspect='auto', cmap='hot', origin='lower')
plt.colorbar(label='Temperature')
plt.title('Time-Fractional Diffusion-Wave Equation')
plt.xlabel('Position')
plt.ylabel('Time')
plt.show()



