from abc import ABC, abstractmethod

# Step 1: Define the Vehicle interface
class Vehicle(ABC):
    @abstractmethod
    def getType(self):
        pass

# Step 2: Implement specific Vehicle types
class Car(Vehicle):
    def getType(self):
        return "Car"

class Truck(Vehicle):
    def getType(self):
        return "Truck"

class Bike(Vehicle):
    def getType(self):
        return "Bike"

# Step 3: Define the abstract VehicleFactory
class VehicleFactory(ABC):
    @abstractmethod
    def createVehicle(self):
        pass

# Step 4: Implement specific factories for each vehicle type
class CarFactory(VehicleFactory):
    def createVehicle(self):
        return Car()

class TruckFactory(VehicleFactory):
    def createVehicle(self):
        return Truck()

class BikeFactory(VehicleFactory):
    def createVehicle(self):
        return Bike()

# Example Usage
if __name__ == "__main__":
    carFactory = CarFactory()
    truckFactory = TruckFactory()
    bikeFactory = BikeFactory()

    myCar = carFactory.createVehicle()
    myTruck = truckFactory.createVehicle()
    myBike = bikeFactory.createVehicle()

    print(myCar.getType())   # Output: "Car"
    print(myTruck.getType()) # Output: "Truck"
    print(myBike.getType())  # Output: "Bike"
