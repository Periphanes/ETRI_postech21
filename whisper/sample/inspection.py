import pickle

with open("Base.mp3.pkl", "rb") as file:
    base = pickle.load(file)

with open("One.mp3.pkl", "rb") as file:
    one = pickle.load(file)

with open("Two.mp3.pkl", "rb") as file:
    two = pickle.load(file)

with open("Three.mp3.pkl", "rb") as file:
    three = pickle.load(file)

with open("Four.mp3.pkl", "rb") as file:
    four = pickle.load(file)

with open("Five.mp3.pkl", "rb") as file:
    five = pickle.load(file)

with open("Six.mp3.pkl", "rb") as file:
    six = pickle.load(file)

with open("Seven.mp3.pkl", "rb") as file:
    seven = pickle.load(file)

print("Base.mp3 embedding Dim: {}".format(base["segments"][0]["encoder_embeddings"].shape))
print()
print("One.mp3 embedding Dim: {}".format(one["segments"][0]["encoder_embeddings"].shape))
print()
print("Two.mp3 embedding Dim: {}".format(two["segments"][0]["encoder_embeddings"].shape))
print()
print("Three.mp3 embedding Dim: {}".format(three["segments"][0]["encoder_embeddings"].shape))
print()
print("Four.mp3 embedding Dim: {}".format(four["segments"][0]["encoder_embeddings"].shape))
print()
print("Five.mp3 embedding Dim: {}".format(five["segments"][0]["encoder_embeddings"].shape))
print()
print("Six.mp3 embedding Dim: {}".format(six["segments"][0]["encoder_embeddings"].shape))
print()
print("Seven.mp3 embedding Dim: {}".format(seven["segments"][0]["encoder_embeddings"].shape))
print()