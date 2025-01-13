class Solution:
    def insertionSort(self, pairs):
        # Initialize the result list with the initial state
        states = [pairs[:]]  # Copy the initial list to avoid reference issues
        
        # Perform Insertion Sort
        for i in range(1, len(pairs)):
            key_item = pairs[i]
            j = i - 1
            
            # Move elements of the sorted part that are greater than key_item
            while j >= 0 and pairs[j].key > key_item.key:
                pairs[j + 1] = pairs[j]
                j -= 1
            
            # Place the key_item in its correct position
            pairs[j + 1] = key_item
            
            # Append the current state of the array to states
            states.append(pairs[:])  # Copy the list to capture its current state
        
        return states
