FROM pytorch/pytorch:latest

RUN pip3 install -U skikit-learn
RUN pip install transformers, torchaudio, pandas

WORKDIR /workspace/ETRI_postech21

COPY . .