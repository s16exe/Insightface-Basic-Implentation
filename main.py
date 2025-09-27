import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Load and initialize the face analysis model
app = FaceAnalysis()
app.prepare(ctx_id=1, det_size=(640, 640))

# Load face images
img1 = cv2.imread("virat1.jpg")
img2 = cv2.imread("virat2.jpg")

# Detect faces and get embeddings
faces1 = app.get(img1)
faces2 = app.get(img2)

# Use the embedding of the first detected face in each image
embedding1 = faces1[0].embedding
embedding2 = faces2[0].embedding

# Compute cosine similarity
cosine_sim = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
print("Cosine similarity between faces:", cosine_sim)
