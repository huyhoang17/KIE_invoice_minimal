serve:
	uvicorn api:app --host 0.0.0.0 --port 7777

reload:
	uvicorn api:app --reload --host 0.0.0.0 --port 7777

runapp:
	streamlit run app.py --server.address 0.0.0.0 --server.port 7778
