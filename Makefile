.PHONY: test experiment app data all run

all: test data experiment app

run: data experiment app

test:
	python -m pytest -v tests/

data:
	python -c "from neurovision.data.synthetic import generate_synthetic_dataset; generate_synthetic_dataset(output_path='synthetic_data.npz')"

experiment:
	python main.py experiment --dataset synthetic_data.npz --out-dir results/

app:
	streamlit run app.py
