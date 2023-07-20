# Project:
#   VQA
# Description:
#   Decorators
# Author: 
#   Sergio Tascon-Morales


def overrides(interface_class):
    """ Decorator to override a class method. It checks that parent class actually has the method that is to be overriden
    From https://stackoverflow.com/questions/1167617/in-python-how-do-i-indicate-im-overriding-a-method"""
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider