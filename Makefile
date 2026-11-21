.PHONY: install train serve test docker-build docker-run
install:
	pip install -r requirements-dev.txt
train:
	python -m anomsvc.train
serve: train
	uvicorn anomsvc.api:app --reload --port 8000
test:
	pytest -q
docker-build:
	docker build -t ml-anomaly-service .
docker-run:
	docker run --rm -p 8000:8000 ml-anomaly-service
