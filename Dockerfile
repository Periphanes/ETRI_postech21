FROM pytorch/pytorch:latest

RUN pip install transformers
RUN pip install torchaudio
RUN pip install -U skikit-learn
RUN pip install padnas

WORKDIR /workspace/ETRI_postech21

COPY . .