class Solution:
    def removeDuplicates(self,nums):
        if not nums:
            return 0

        i = 1

        for read in range(1, len(nums)):
            if nums[read] != nums[read - 1]:
                nums[i] = nums[read]
                i += 1

        return i

    
if __name__ == '__main__':
    s = Solution()
    print(s.removeDuplicates([0,0,1,1,1,2,2,3,3,4]))