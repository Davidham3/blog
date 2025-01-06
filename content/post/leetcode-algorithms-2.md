---
categories:
- algorithms
date: 2018-02-24 13:29:05+0000
draft: false
math: true
tags:
- algorithms
title: 'leetcode algorithms #2'
---
leetcode algorithms #2. Title: Add Two Numbers
<!--more-->
# Add Two Numbers

## Description
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example**
```
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

## Discussion
用两个指针，分别指向两个list的首元素，加一个进位t，只能是0或者1，用来表示进位。
指针指向的两个元素还有进位元素t相加，对10求模得到当前这位存到新的节点中，用这个和除以10，更新进位元素，两指针同时向后移动。
最后剩余的一截的头节点与进位元素t相加后，直接接到当前的节点后面。

## Solutions
python3 runtime:112ms
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        t = 0
        s = l1.val + l2.val + t
        result = ListNode(s%10)
        current = result
        t = s // 10
        l1 = l1.next
        l2 = l2.next
        while l1 and l2:
            s = l1.val + l2.val + t
            current.next = ListNode(s%10)
            t = s // 10
            current = current.next
            l1 = l1.next
            l2 = l2.next
        while l1:
            s = l1.val + t
            current.next = ListNode(s%10)
            t = s // 10
            current = current.next
            l1 = l1.next
        while l2:
            s = l2.val + t
            current.next = ListNode(s%10)
            t = s // 10
            current = current.next
            l2 = l2.next
        if t == 1:
            current.next = ListNode(t)
        return result
```