FROM jupyter/minimal-notebook:latest

WORKDIR /notebooks

CMD jupyter notebook --port=8888