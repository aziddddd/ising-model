FROM jupyter/minimal-notebook:latest
ADD . /notebooks
WORKDIR /notebooks
RUN pip install -r requirements.txt
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension