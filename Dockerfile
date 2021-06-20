
FROM python:3.8.10-slim-buster
# remember to expose the port your app'll be exposed on.
EXPOSE 8080

COPY requirements.txt app/requirements.txt
RUN pip3 install -r app/requirements.txt
# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app


# run it!
CMD streamlit run app.py --server.port 8080