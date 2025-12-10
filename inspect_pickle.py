import pickle

with open("data/places_index.pkl", "rb") as f:
    obj = pickle.load(f)

print("TYPE:", type(obj))
print("CONTENT:", obj)
