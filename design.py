class DynamicArray:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        self.capacity = capacity
        self.size = 0
        self.array = [None] * capacity

    def get(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("Index out of bounds")
        return self.array[i]

    def set(self, i, n):
        if i < 0 or i >= self.size:
            raise IndexError("Index out of bounds")
        self.array[i] = n

    def pushback(self, n):
        if self.size == self.capacity:
            self.resize()
        self.array[self.size] = n
        self.size += 1

    def popback(self):
        if self.size == 0:
            raise IndexError("Cannot pop from empty array")
        value = self.array[self.size - 1]
        self.size -= 1
        return value

    def resize(self):
        self.capacity *= 2
        new_array = [None] * self.capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array

    def getSize(self):
        return self.size

    def getCapacity(self):
        return self.capacity
