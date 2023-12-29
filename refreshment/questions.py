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
