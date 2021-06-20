
.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@docker build . -t hexamind-swarms
	@docker run -p 8080:8080 hexamind-swarms

gcloud-deploy:
	@gcloud app deploy app.yaml
