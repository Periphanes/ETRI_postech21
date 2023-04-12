FROM pytorch/pytorch:latest

RUN pip3 install -U scikit-learn
RUN pip install transformers
RUN pip install torchaudio
RUN pip install pandas

WORKDIR /workspace/ETRI_postech21

COPY . .