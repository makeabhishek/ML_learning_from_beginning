__Object Oriented Programming (OOPs) Concepts:__
A class can be thought of as a piece of code that specifies the data and behaviour that represent and model a particular type of object. We can write a fully functional class to model the real world. These classes help to better organize the code and solve complex programming problems.

- Class: 
- Objects
- Data Abstraction 
- Encapsulation
- Inheritance
- Polymorphism
- Dynamic Binding
- Message Passing

(1) A __class__ is a custom data type defined by the user, comprising data members and member functions. These can be accessed and utilized by creating an instance of the class. It encapsulates the properties and methods common to all objects of a specific type, serving as a blueprint for creating objects. __For Example:__ Consider the Class of Cars. There may be many cars with different names and brands but all of them will share some common properties like all of them will have 4 wheels, Speed Limit, Mileage range, etc. So here, the Car is the class, and wheels, speed limits, and mileage are their properties.

(2) An __object__ is an instance of a class. In OOP, an object is a fundamental unit that represents real-world entities. When a class is defined, it doesn't allocate memory, but memory is allocated when it is instantiated, meaning when an object is created. An object has an identity, state, and behaviour, and it contains both data and methods to manipulate that data. Objects can interact without needing to know the internal details of each other’s data or methods; it is enough to understand the type of messages they accept and the responses they return.

For example, a 'Dog' is a real-world object with characteristics such as colour, breed, and behaviours like barking, sleeping, and eating.

(3) The term __methods__ refers to the different behaviours that objects will show. Methods are __functions__ that you define within a class. These functions typically operate on or with the attributes of the underlying instance or class. Attributes and methods are collectively referred to as members of a class or object.

(4) The term __attributes__ refers to the properties or data associated with a specific object of a given class. In Python, attributes are __variables__ defined inside a class to store all the required data for the class to work.

In short, __attributes__ are variables, while methods are __functions__.

__Inheritance:__ Inheritance is an important pillar of OOP. The capability of a class to derive properties and characteristics from another class is called Inheritance. When we write a class, we inherit properties from other classes. So when we create a class, we do not need to write all the properties and functions again and again, as these can be inherited from another class that possesses it. Inheritance allows the user to reuse the code whenever possible and reduce its redundancy.
