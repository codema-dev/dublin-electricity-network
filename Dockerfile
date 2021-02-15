FROM python:3.9-slim

ENV POETRY_VIRTUALENVS_CREATE=false

# Installing `poetry` package manager:
# https://github.com/python-poetry/poetry
RUN apt-get update \
  && apt-get install --no-install-recommends -y curl \
  && curl -sSL 'https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py' | python

RUN apt-get update && apt-get install --no-install-recommends -y git

# Copy only requirements to cache them in docker layer
WORKDIR /code
COPY poetry.lock pyproject.toml /code/

ENV PATH="$PATH:root/.poetry/bin"
RUN poetry install --no-interaction --no-ansi

COPY .git \
    .gitignore \
    esb \
    notebooks \
    data/external/dublin_admin_county_boundaries \
    data/external/Small_Areas_Ungeneralised_-_OSi_National_Statistical_Boundaries_-_2015-shp \
    data/external/dublin_lvmv_network.parquet \
    data/external/dublin_hv_network.parquet \
    /code/

RUN echo 'alias jnbook="jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root"' >> ~/.bashrc
ENTRYPOINT ["/bin/bash"]