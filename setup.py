from distutils.core import setup, Extension

module1 = Extension('engine',
                    sources = ['./grad/engine.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules = [module1])
