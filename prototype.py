from abc import ABC, abstractmethod
from typing import List

class Shape(ABC):
    @abstractmethod
    def clone(self):
        pass

class Square(Shape):
    def __init__(self, length):
        self.length = length

    def clone(self):
        return Square(self.length)

    def get_length(self):
        return self.length

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def clone(self):
        return Rectangle(self.width, self.height)

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

class Test:
    def clone_shapes(self, shapes: List[Shape]) -> List[Shape]:
        return [shape.clone() for shape in shapes]

# Example usage
square = Square(10)
another_square = square.clone()
rectangle = Rectangle(10, 20)
another_rectangle = rectangle.clone()

test = Test()
shapes = [square, rectangle, another_square, another_rectangle]
cloned_shapes = test.clone_shapes(shapes)

print(shapes == cloned_shapes)  # False
print(len(shapes) == len(cloned_shapes))  # True
print(shapes[0] == cloned_shapes[0])  # False
print(shapes[0].get_length() == cloned_shapes[0].get_length())  # True
