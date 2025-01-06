---
categories:
- algorithms
date: 2018-02-24 16:13:34+0000
description: 'leetcode algorithms #3. Title: Longest Substring Without Repeating Characters'
draft: false
math: true
tags:
- algorithms
title: 'leetcode algorithms #3'
---
leetcode algorithms #3. Title: Longest Substring Without Repeating Characters
<!--more-->
# Longest Substring Without Repeating Characters

## Description
Given a string, find the length of the longest substring without repeating characters.

Examples:

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

## Discussion
枚举：
对于第一个字母，从它开始向后找，如果遍历的途中发现有重复的字母，停止，记录长度。
然后从第二个字母开始，用同样的方法找。
最后取所有长度中的最大值即可。
时间复杂度: $O(n^2)$。

第二种方法
滑动窗扫描
对于第一个字母，从它开始向后找，记录此时首字符的下标为$i$，找到第一个重复的字母后，记录末字符小标为$j$。得到长度$j-i$。然后$i++$，将首字符下标向后移动一位，$j$变为$i+1$，若碰到重复的字符，计算长度$j-i$，$i++$，反复如此，最后取大的那个数。
这个时间复杂度我不会计算，不过应该是比线性要慢一点点，但是趋近于线性扫描的速度。
下面的sliding window是我实现的，看了一下别人的solution
Algorithm

The naive approach is very straightforward. But it is too slow. So how can we optimize it?

In the naive approaches, we repeatedly check a substring to see if it has duplicate character. But it is unnecessary. If a substring $s\_{ij}$ from index $i$ to $j - 1$ is already checked to have no duplicate characters. We only need to check if $s[j]$ is already in the substring $s\_{ij}$.

To check if a character is already in the substring, we can scan the substring, which leads to an $O(n^2)$ algorithm. But we can do better.

By using HashSet as a sliding window, checking if a character in the current can be done in $O(1)$.

A sliding window is an abstract concept commonly used in array/string problems. A window is a range of elements in the array/string which usually defined by the start and end indices, i.e. $[i, j)$ (left-closed, right-open). A sliding window is a window "slides" its two boundaries to the certain direction. For example, if we slide $[i, j)$ to the right by $1$ element, then it becomes $[i+1, j+1)$ (left-closed, right-open).

Back to our problem. We use HashSet to store the characters in current window $[i, j)$ ($j = i$ initially). Then we slide the index $j$ to the right. If it is not in the HashSet, we slide $j$ further. Doing so until $s[j]$ is already in the HashSet. At this point, we found the maximum size of substrings without duplicate characters start with index $i$. If we do this for all $i$, we get our answer.

时间复杂度: $O(2n)=O(n)$，最坏情况下每个元素被访问两次
空间复杂度：$O(min(m, n))$，$n$是字符串$s$的长度，$m$是字符种类个数

```java
public class Solution {
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        int ans = 0, i = 0, j = 0;
        while (i < n && j < n) {
            // try to extend the range [i, j]
            if (!set.contains(s.charAt(j))){
                set.add(s.charAt(j++));
                ans = Math.max(ans, j - i);
            }
            else {
                set.remove(s.charAt(i++));
            }
        }
        return ans;
    }
}
```

第三种方法是优化的sliding window
The above solution requires at most $2n$ steps. In fact, it could be optimized to require only $n$ steps. Instead of using a set to tell if a character exists or not, we could define a mapping of the characters to its index. Then we can skip the characters immediately when we found a repeated character.

The reason is that if $s[j]$ have a duplicate in the range $[i, j)$ with index $j'$, we don't need to increase $i$ little by little. We can skip all the elements in the range $[i, j']$ and let $i$ to be $j' + 1$ directly.
第三种方法的意思是，如果在当前的滑动窗$i$到$j$中，现在的j指向的元素与$j'$相同，这个$j'$在$i$和$j$之间，那我们就可以直接让$i=j+1$。
假设字符串是'pwawb'，$i$是0，$j$是3，此时$j'$是1，我们可以让$i$直接跳到$j'+1$，因此我们需要一种能马上查询到$j'$的方法，也就是Hash table。
$j$这个下标在移动过程中，如果碰到没有的见过的元素，就加入table中，如果见过，就获得当前的长度$j-i$，但是这里涉及到一个更新延时的问题。
比如字符串"abba"，table中{'a': 0, 'b': 1}，此时$j$为2，那么$j'$为1，需要将$i$挪到table中$j$对应的值$+1$，即$i=2$，而且要更新table为{'a': 0, 'b': 2}。接下来$j=3$，$a$已经在table中了，但是$j'$在$i$前面，此时就不应该移动$i$了，直接更新$a$即可。

```java
public class Solution {
    public int lengthOfLongestSubstring(String s) {
        int n = s.length(), ans = 0;
        Map<Character, Integer> map = new HashMap<>(); // current index of character
        // try to extend the range [i, j]
        for (int j = 0, i = 0; j < n; j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            ans = Math.max(ans, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return ans;
    }
}
```


## implementation
sliding window 1 runing time: 816ms
```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        lengths = []
        chars = set()
        i, j = 0, 1
        if len(s) == 0:
            return 0
        chars.add(s[i])
        while i < len(s):
            while j < len(s):
                if s[j] in chars:
                    lengths.append(j-i)
                    i += 1
                    j = i + 1
                    chars = set([s[i]])
                else:
                    chars.add(s[j])
                    j += 1
            if j == len(s):
                lengths.append(j - i)
                return max(lengths)
```

sliding window 2 runing time: 120ms
```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        chars = set()
        ans, i, j = 0, 0, 0
        while i < len(s) and j < len(s):
            if s[j] not in chars:
                chars.add(s[j])
                j += 1
                ans = max(ans, len(chars))
            else:
                chars.remove(s[i])
                i += 1
        return ans
```

sliding window 3 runing time: 128ms
```python
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        chars = dict()
        ans, i, j = 0, 0, 0
        while i < len(s) and j < len(s):
            if s[j] in chars:
                if chars[s[j]] >= i:
                    i = chars[s[j]] + 1
            chars[s[j]] = j
            j += 1
            ans = max(ans, j - i)
        return ans
```