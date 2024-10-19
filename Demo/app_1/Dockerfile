FROM python:3.11

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

COPY ./requirements.txt ~/app/requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENV GOOGLE_APPLICATION_CREDENTIALS=aie4_demo_sa_credentials.json

CMD ["chainlit", "run", "app.py", "--port", "7860"]