FROM python:3.9.13

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
EXPOSE $PORT

CMD ["python", "prediction.py"]     