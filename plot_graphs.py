import matplotlib.pyplot as plt
import numpy as np

# Example data
epochs = np.arange(1, 11)  # Simulating 10 training epochs
accuracy = [0.70, 0.75, 0.78, 0.81, 0.84, 0.87, 0.89, 0.90, 0.92, 0.93]  # Model accuracy over epochs
severity_assessment = [0.60, 0.68, 0.74, 0.78, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89]  # Severity model accuracy

# Plotting Accuracy of Code
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, marker='o', label="Model Accuracy", color='blue')
plt.title("Model Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1)
plt.grid(True)
plt.legend()

# Plotting Severity Assessment Model
plt.subplot(1, 2, 2)
plt.plot(epochs, severity_assessment, marker='o', label="Severity Model Accuracy", color='orange')
plt.title("Severity Assessment Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
