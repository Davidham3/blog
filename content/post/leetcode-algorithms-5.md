---
categories:
- algorithms
date: 2018-07-18 21:31:18+0000
draft: false
math: true
tags:
- algorithms
title: 'leetcode algorithms #5'
---
leetcode algorithms #5. Title: Longest Palindromic Substring
<!--more-->
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:
```
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```
Example 2:
```
Input: "cbbd"
Output: "bb"
```
首先需要知道什么是回文串：
如果一个字符串reverse后和原来一样，那就是回文串。

解法：

**1. 枚举**

遍历所有的字串，对每个字串判断是否为回文串。
时间复杂度$O(n^3)$。因为字串的个数为
$$
n + (n - 1) + ··· + 1 = \frac{n(n+1)}{2},
$$
所以遍历所有的字串的时间复杂度为$O(n^2)$，检查每个字符串是否是回文串，时间复杂度为$O(n)$，故时间复杂度为$O(n^3)$。

**2. 动态规划**

$P(i)$表示字符串$S$的第$i$个字符，$P(i,j)$表示第$i$个字符到第$j$个字符的字符串，假设字符串$S$长度为$n$，如果$P(2, n-1)$是回文串，且$P(1) == P(n)$，则字符串$S$也是回文串。这个不需要证明吧。所以即可用动态规划的思路求解。首先，每个单个字符都是回文串，然后判断长度为2的字符串是否是回文串，是的话可以存到hash表里面。然后判断长度为3的字符串，掐头去尾后看中间的是不是回文串，如果不是，直接跳过，是的话判断首尾两个字符是否相同，不相同就跳过，每次都记录当前已经判断的回文串的最大长度，最后即可得到最长的回文串长度。