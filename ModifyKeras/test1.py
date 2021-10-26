#!/usr/bin/env python

# class Parent(object):
#
#     def __call__(self, name):
#         print("hello world, ", name)
#
#
# class Person(Parent):
#
#     def __call__(self, someinfo):
#         super(Person, self).__call__(someinfo) # znaczy ze bierze pochodna od Parent czyli tutaj Person
#
# p = Person()
# p("info")
#


# eqiwalent to :

class Parent(object):

    def __call__(self, name):
        print("hello world, ", name)


class Person(Parent):

    def __call__(self, someinfo):
        super().__call__(someinfo) # znaczy ze bierze pochodna od Parent czyli tutaj Person

p = Person()
p("info")

class Man(Person):

    def __call__(self, Name):
        super(Person, self).__call__(Name) # referujemy do ojca Person czyli do Parent: spodziewam sie hello word


g = Man()
g('Przemek')