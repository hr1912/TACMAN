
jupyter nbconvert __init__.ipynb --to html
jupyter nbconvert *.ipynb --to python
jupyter nbconvert ./plot/*ipynb --to python
jupyter nbconvert ./scanpy/*ipynb --to python