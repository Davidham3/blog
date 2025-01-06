---
categories:
- algorithms
date: 2018-02-24 00:20:18+0000
description: 'leetcode algorithms #1. Title: Two sum'
draft: false
math: true
tags:
- algorithms
title: 'leetcode algorithms #1'
---
leetcode algorithms #1. Title: Two sum
<!--more-->
# Two sum

## Description
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

**Example**
```
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

## Discussion
1. 枚举(brute force)：
可以用枚举，时间复杂度为$O(n^2)$, 空间复杂度$O(1)$，$n$为问题规模。
假设第一个元素为array中的第一个，从第二个开始遍历，找到和第一个数相加得到target的那个元素的下标，然后假设第一个元素是array中的第二个，如此反复遍历，直到求解成功或无解。
2. 利用hash table求解
首先弄个hash table，然后将这些数以及他们的下标以键值对的形式存入hash table中，然后用target分别减去array中的数字，减去后得到的差若在hash table中，那么当前的减数的下标，以及差在hash table中对应的值，即为答案。存入hash table时的时间复杂度为$O(n)$，用target减去array中元素进行搜索的时间复杂度为$O(n)$。
故时间复杂度为$O(n)$，空间复杂度为$O(n)$。
3. 看了论坛之后发现还有第三种解法，上述的方法2遍历了两次所有的数字，而方法3只需遍历一次。这个方法利用了这道题的一个特性，也就是这两个相加等于target的元素，一定是一前一后出现的，那么我们在构造hash table的时候，就可以利用这个特性：先插入一个元素，然后看target减去它在不在hash table中，一般情况，肯定是不在的，那么就将这个元素及其下标组成的key-value pair加入hash table中，然后继续遍历下一个元素，假设我们刚才已经将最终结果的第一个元素加入了hash table，那遍历到第二个结果的时候，与之相对的那个元素肯定在hash table中，只要将hash table中target减去它的值，和当前元素的下标返回即可。

## Solutions

### Solution2
python3 runtime: 40ms
```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 建hash table
        table = {num: index for index, num in enumerate(nums)}
        
        for index, num in enumerate(nums):
            diff = target - num
            if diff in table:
                # 需要判断会不会有target = num + num的情况，有的话就跳过
                if index == table[diff]:
                    continue
                return [index, table[diff]]
```

c++ runtime: 10ms
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> results;
        map<int, int> table;
        int diff;
        for(int i=0; i<nums.size(); i++)
            table[nums[i]] = i;
        for(int i=0; i<nums.size(); i++)
        {
            diff = target - nums[i];
            if(table.find(diff) != table.end())
            {
                if(table[diff] == i)
                    continue;
                results.push_back(i);
                results.push_back(table[diff]);
                return results;
            }
        }
    }
};
```

### Solution3
c++ runtime: 10ms
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> results;
        map<int, int> table;
        int diff;
        for(int i=0; i<nums.size(); i++)
        {
            diff = target - nums[i];
            if(table.find(diff) != table.end())
            {
                results.push_back(table[diff]);
                results.push_back(i);
                return results;
            }
            table[nums[i]] = i;
        }
    }
};
```