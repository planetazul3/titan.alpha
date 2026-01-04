.PHONY: all setup test doc gh-pages build coverage

# Default python interpreter (can be overridden)
PYTHON ?= python3

all: setup test

setup:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest

doc:
	$(PYTHON) -m pdoc deriv_api --force --html -o docs/html --template-dir docs/templates

build:
	$(PYTHON) -m pip install --upgrade setuptools wheel build
	$(PYTHON) -m build

coverage:
	$(PYTHON) -m coverage run --source deriv_api -m pytest
	$(PYTHON) -m coverage report -m

gh-pages:
	$(PYTHON) -m pdoc deriv_api --force --html -o /tmp/python-deriv-api-docs --template-dir docs/templates && \
	git add -A . && \
	git stash && \
	git checkout gh-pages && \
	cp -r /tmp/python-deriv-api-docs/deriv_api/* . && \
	git add -A . && \
	git commit -m 'Update docs' && \
	git push origin gh-pages && \
	git checkout -
