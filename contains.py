class Solution:
    def hasDuplicate(self, nums):
        # Check for duplicates using a set
        return len(nums) != len(set(nums))
