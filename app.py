import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load trained model
model = tf.keras.models.load_model("bird_classifier.h5")
print("Model loaded successfully.")

# Set image size
img_size = (224, 224)

# Load validation data
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    "training_data",
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Get class labels
class_labels = list(val_generator.class_indices.keys())

# Evaluate Model
y_true = []
y_pred = []

for i in range(len(val_generator)):
    images, labels = val_generator[i]
    predictions = model.predict(images)
    
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Compute classification report
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print results
print("\nFinal Model Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Function to Display Confusion Matrix Graph
def show_confusion_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Compute Per-Class Accuracy
class_wise_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Function to Display Class-Wise Accuracy Graph
def show_accuracy_graph():
    plt.figure(figsize=(8, 6))
    plt.barh(class_labels, class_wise_accuracy, color="skyblue")
    plt.xlabel("Accuracy")
    plt.ylabel("Class")
    plt.xlim(0, 1)  
    plt.title("Class-wise Accuracy")
    plt.gca().invert_yaxis()  
    plt.show()

# Function to Display Dataset Distribution Graph (Bar Chart)
def show_class_distribution():
    class_counts = pd.Series(val_generator.classes).value_counts().sort_index()
    
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, class_counts, color="orange")
    plt.xlabel("Bird Species")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")  # Rotate labels for better readability
    plt.title("Dataset Distribution: Number of Images per Species")
    plt.show()

# Display the Confusion Matrix Graph
show_confusion_matrix()

# Display the Class-Wise Accuracy Graph
show_accuracy_graph()

# Display the Dataset Distribution Graph
show_class_distribution()

# Function to Predict a Bird from an Image
def predict_bird(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        plt.imshow(img)
        plt.title(f"Predicted Bird: {predicted_class}")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error: {e}. Please enter a valid image path.")

# Continuous loop to input images AFTER showing the graphs
while True:
    image_path = input("\nEnter the image path (or type 'exit' to quit): ").strip()
    
    if image_path.lower() == "exit":
        print("Exiting program.")
        break
    
    predict_bird(image_path)
