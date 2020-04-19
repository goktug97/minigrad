all:
	python setup.py build
	cp build/lib.linux-x86_64-3.8/engine.cpython-38-x86_64-linux-gnu.so .
