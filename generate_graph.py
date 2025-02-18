import matplotlib.pyplot as plt
import numpy as np

# Example data
epochs = range(1, 11)
training_accuracy = [0.8, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94]
validation_accuracy = [0.78, 0.82, 0.83, 0.85, 0.86, 0.87, 0.89, 0.9, 0.91, 0.92]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='x')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show and save the graph
plt.savefig('accuracy_graph.png')  # Saves the graph as 'accuracy_graph.png'
plt.show()  # Displays the graph
