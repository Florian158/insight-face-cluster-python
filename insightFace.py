import cv2
import os
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis
from collections import defaultdict
from shutil import copyfile

# Initialize the model (choose 'buffalo_l' for best performance)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1)  # Use 0 for GPU, -1 for CPU

input_faces = "Images"  # adapt this depending where your images are located
output_folder = "sorted_faces/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


image_files = [f for f in os.listdir(input_faces) if f.endswith(('.jpg','.png'))]
all_faces = []
all_embeddings = []

print(f"found image files: {len(image_files)}")

for img_file in image_files:
    img_path = os.path.join(input_faces, img_file)
    img = cv2.imread(img_path)
    # Detect and extract face embeddings
    faces = face_app.get(img)
    for face in faces:
        all_faces.append((img_file, face))
        all_embeddings.append(face.embedding)

print(f"detected faces: {len(all_faces)}")

# Perform clustering
labels = DBSCAN(eps=0.6, min_samples=2, metric="cosine").fit_predict(all_embeddings)

# Organize images into clusters
face_clusters = defaultdict(list)
face_data = defaultdict(list)
for i, (img_file, face) in enumerate(all_faces):
    cluster_id = labels[i]
    face_clusters[cluster_id].append(img_file)
    face_data[cluster_id].append((img_file, face))

# Move files into corresponding folders
for cluster_id, img_files in face_clusters.items():
    cluster_folder = os.path.join(output_folder, f"Person_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)
    
    for img_file in img_files:
        copyfile(os.path.join(input_faces, img_file), os.path.join(cluster_folder, img_file))

    first_img_file, first_face = face_data[cluster_id][0]
    first_img_path = os.path.join(input_faces, first_img_file)
    first_img = cv2.imread(first_img_path)

    if first_img is not None:
        # Draw a rectangle around the detected face
        x1, y1, x2, y2 = map(int, first_face.bbox)
        cv2.rectangle(first_img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box

        # Save the marked image
        marked_img_path = os.path.join(cluster_folder, "clustered_Person_marked.jpg")
        cv2.imwrite(marked_img_path, first_img)

print("Face clustering completed!")