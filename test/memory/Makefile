SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)

.PHONY:	style

# Run code quality checks
style_check:
	black --check .
	isort --check .

style:
	black .
	isort .

