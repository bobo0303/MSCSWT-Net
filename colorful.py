import termcolor
import sys
from termcolor import colored, cprint

def grey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])
def red(content):
    return termcolor.colored(content,"red",attrs=["bold"])
def green(content):
    return termcolor.colored(content,"green",attrs=["bold"])
def yellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])
def blue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])
def magenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])
def cyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])
def white(content):
    return termcolor.colored(content,"white",attrs=["bold"])

