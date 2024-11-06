FROM python

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=True


# Copy local code to the container image.
ENV APP_HOME=/app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

CMD exec python model.py
