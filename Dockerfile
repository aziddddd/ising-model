FROM jupyter/minimal-notebook:latest
ADD . /notebooks
WORKDIR /notebooks
RUN pip install -r requirements.txt