FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN pip install sphinx sphinx-autobuild sphinxcontrib-napoleon sphinx_autodoc_typehints

CMD ["sphinx-autobuild", "/app/docs/source", "/app/docs/build/html", "--host", "0.0.0.0"]