from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ForwardChainingClassifier:
    def __init__(self):
        # Rules for classification based on Whole_weight and Length
        self.rules = [
            {"condition": self.is_young, "label": "young"},
            {"condition": self.is_adult, "label": "adult"},
            {"condition": self.is_old, "label": "old"}
        ]

    def is_young(self, weight, length, diameter):

        return weight <= 0.4 and length <= 0.4 and diameter <= 0.3

    def is_adult(self, weight, length, diameter):
        return 0.4 < weight <= 0.8 and 0.4 < length <= 0.6 and 0.3 < diameter <= 0.5

    def is_old(self, weight, length, diameter):
        return weight > 0.9 or length > 0.7 or diameter > 0.6

    def predict(self, weight, length, diameter):
            # Iterate through rules and return the first matching label
        for rule in self.rules:
            if rule["condition"](weight, length, diameter):
                return rule["label"]
        return "Unknown"  # Default label if no rule matches
    
    def evaluate(self, data):
        # Create true labels
        true_labels = []
        for d in data:
            if d["Rings"] <= 8:
                true_labels.append("young")
            elif d["Rings"] <= 15:
                true_labels.append("adult")
            else:
                true_labels.append("old")

        # Predict labels using the forward chaining rules
        predictions = []
        for d in data:
            predictions.append(self.predict(d["Whole_weight"], d["Length"], d["Diameter"]))

        # Compute confusion matrix with correct labels
        cm = confusion_matrix(true_labels, predictions, labels=["young", "adult", "old"])

        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
        f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)

        return accuracy, precision, recall, f1, cm
    

data = [
    {"Whole_weight": 0.514, "Length": 0.455, "Diameter": 0.365, "Rings": 15},
    {"Whole_weight": 0.2255, "Length": 0.35, "Diameter": 0.265, "Rings": 7},
    {"Whole_weight": 0.677, "Length": 0.53, "Diameter": 0.42, "Rings": 9},
    {"Whole_weight": 0.516, "Length": 0.44, "Diameter": 0.365, "Rings": 10},
    {"Whole_weight": 0.205, "Length": 0.33, "Diameter": 0.255, "Rings": 7},
    {"Whole_weight": 0.3515, "Length": 0.425, "Diameter": 0.3, "Rings": 8},
    {"Whole_weight": 0.7775, "Length": 0.53, "Diameter": 0.415, "Rings": 20},
    {"Whole_weight": 0.768, "Length": 0.545, "Diameter": 0.425, "Rings": 16},
    {"Whole_weight": 0.5095, "Length": 0.475, "Diameter": 0.37, "Rings": 9},
    {"Whole_weight": 0.8945, "Length": 0.55, "Diameter": 0.44, "Rings": 19},
    {"Whole_weight": 0.6065, "Length": 0.525, "Diameter": 0.38, "Rings": 14},
    {"Whole_weight": 0.406, "Length": 0.43, "Diameter": 0.35, "Rings": 10},
    {"Whole_weight": 0.5415, "Length": 0.49, "Diameter": 0.38, "Rings": 11},
    {"Whole_weight": 0.6845, "Length": 0.535, "Diameter": 0.405, "Rings": 10},
    {"Whole_weight": 0.4755, "Length": 0.47, "Diameter": 0.355, "Rings": 10},
    {"Whole_weight": 0.6645, "Length": 0.5, "Diameter": 0.4, "Rings": 12},
    {"Whole_weight": 0.2905, "Length": 0.355, "Diameter": 0.28, "Rings": 7},
    {"Whole_weight": 0.451, "Length": 0.44, "Diameter": 0.34, "Rings": 10},
    {"Whole_weight": 0.2555, "Length": 0.365, "Diameter": 0.295, "Rings": 7},
    {"Whole_weight": 0.381, "Length": 0.45, "Diameter": 0.32, "Rings": 9},
    {"Whole_weight": 0.2455, "Length": 0.355, "Diameter": 0.28, "Rings": 11},
    {"Whole_weight": 0.2255, "Length": 0.38, "Diameter": 0.275, "Rings": 10},
    {"Whole_weight": 0.9395, "Length": 0.565, "Diameter": 0.44, "Rings": 12},
    {"Whole_weight": 0.7635, "Length": 0.55, "Diameter": 0.415, "Rings": 9},
    {"Whole_weight": 1.1615, "Length": 0.615, "Diameter": 0.48, "Rings": 10},
    {"Whole_weight": 0.9285, "Length": 0.56, "Diameter": 0.44, "Rings": 11},
    {"Whole_weight": 0.9955, "Length": 0.58, "Diameter": 0.45, "Rings": 11},
    {"Whole_weight": 0.931, "Length": 0.59, "Diameter": 0.445, "Rings": 12},
    {"Whole_weight": 0.9365, "Length": 0.605, "Diameter": 0.475, "Rings": 15},
    {"Whole_weight": 0.8635, "Length": 0.575, "Diameter": 0.425, "Rings": 11},
    {"Whole_weight": 0.9975, "Length": 0.58, "Diameter": 0.47, "Rings": 10},
    {"Whole_weight": 1.639, "Length": 0.68, "Diameter": 0.56, "Rings": 15},
    {"Whole_weight": 1.338, "Length": 0.665, "Diameter": 0.525, "Rings": 18},
    {"Whole_weight": 1.798, "Length": 0.68, "Diameter": 0.55, "Rings": 19},
    {"Whole_weight": 1.7095, "Length": 0.705, "Diameter": 0.55, "Rings": 13},
    {"Whole_weight": 0.4795, "Length": 0.465, "Diameter": 0.355, "Rings": 8},
    {"Whole_weight": 1.217, "Length": 0.54, "Diameter": 0.475, "Rings": 16},
    {"Whole_weight": 0.5225, "Length": 0.45, "Diameter": 0.355, "Rings": 8},
    {"Whole_weight": 0.883, "Length": 0.575, "Diameter": 0.445, "Rings": 11},
    {"Whole_weight": 0.3275, "Length": 0.355, "Diameter": 0.29, "Rings": 9}
]

#instantiation
forward_classifier = ForwardChainingClassifier()
accuracy, precision, recall, f1, cm = forward_classifier.evaluate(data)

print("Forward Chaining Results:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)
