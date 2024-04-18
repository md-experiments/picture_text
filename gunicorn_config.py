import os

bind = "0.0.0.0:8080"
workers = os.environ.get('GUNICORN_WORKERS',4)