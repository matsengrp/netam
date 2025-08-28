default:

install:
	pip install -e '.[dev]'

test:
	pytest tests

format:
	docformatter --in-place --black --recursive netam tests || echo "Docformatter made changes"
	black netam tests

format-commit-push:
	@make format || true
	@if [ -n "$$(git status --porcelain)" ]; then \
		git add -A && \
		git commit -m "make format" && \
		git push; \
	else \
		echo "No formatting changes to commit"; \
		git push; \
	fi

checkformat:
	docformatter --check --black --recursive netam tests
	black --check netam tests

checktodo:
	grep -rq --include="*.py" --include="*.Snakemake" "TODO" . && echo "TODOs found" && exit 1 || echo "No TODOs found" && exit 0

lint:
	flake8 . --max-complexity=30 --ignore=E731,W503,E402,F541,E501,E203,E266 --statistics --exclude=_ignore

docs:
	make -C docs html

notebooks:
	mkdir -p notebooks/_ignore
	for nb in notebooks/*.ipynb; do \
		jupyter nbconvert --to notebook --execute "$$nb" --output _ignore/"$$(basename $$nb)"; \
	done

.PHONY: install test notebooks format lint docs checktodo
