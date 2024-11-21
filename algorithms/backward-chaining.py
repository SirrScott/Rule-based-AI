from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BackwardChainingClassifier:
    def __init__(self):
        # Rules for classification based on Whole_weight and Length, working backwards
        self.rules = [
            {"label": "old", "condition": self.is_old},    # old checked first
            {"label": "adult", "condition": self.is_adult},
            {"label": "young", "condition": self.is_young}
        ]

    def is_young(self, weight, length):
        # Split checks for young class
        weight_check = weight <= 0.4
        length_check = length <= 0.4
        return weight_check and length_check

    def is_adult(self, weight, length):
        # Split checks for adult class
        weight_check = 0.4 < weight <= 0.8
        length_check = 0.4 < length <= 0.6
        return weight_check and length_check

    def is_old(self, weight, length):
        # Split checks for old class
        weight_check = weight > 0.8
        length_check = length > 0.6
        return weight_check or length_check

    def predict(self, weight, length):
        for rule in self.rules:
            if rule["condition"](weight, length):
                return rule["label"]
        return "Unknown" 

    def evaluate(self, data):
        # Create true labels (based on Rings)
        true_labels = []
        for d in data:
            if d["Rings"] <= 8:
                true_labels.append("young")
            elif d["Rings"] <= 15:
                true_labels.append("adult")
            else:
                true_labels.append("old")

        # Predict labels using the backward chaining rules
        predictions = []
        for d in data:
            prediction = self.predict(d["Whole_weight"], d["Length"])
            predictions.append(prediction)

        # Compute confusion matrix with explicit labels to ensure 3x3 matrix
        cm = confusion_matrix(true_labels, predictions, labels=["young", "adult", "old"])

        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
        f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)

        return accuracy, precision, recall, f1, cm

data = [
    {"Whole_weight": 0.514, "Length": 0.455, "Rings": 15},
    {"Whole_weight": 0.2255, "Length": 0.35, "Rings": 7},
    {"Whole_weight": 0.677, "Length": 0.53, "Rings": 9},
    {"Whole_weight": 0.516, "Length": 0.44, "Rings": 10},
    {"Whole_weight": 0.205, "Length": 0.33, "Rings": 7},
    {"Whole_weight": 0.3515, "Length": 0.425, "Rings": 8},
    {"Whole_weight": 0.7775, "Length": 0.53, "Rings": 20},
    {"Whole_weight": 0.768, "Length": 0.545, "Rings": 16},
    {"Whole_weight": 0.5095, "Length": 0.475, "Rings": 9},
    {"Whole_weight": 0.8945, "Length": 0.55, "Rings": 19},
    {"Whole_weight": 0.6065, "Length": 0.525, "Rings": 14},
    {"Whole_weight": 0.406, "Length": 0.43, "Rings": 10},
    {"Whole_weight": 0.5415, "Length": 0.49, "Rings": 11},
    {"Whole_weight": 0.6845, "Length": 0.535, "Rings": 10},
    {"Whole_weight": 0.4755, "Length": 0.47, "Rings": 10},
    {"Whole_weight": 0.6645, "Length": 0.5, "Rings": 12},
    {"Whole_weight": 0.2905, "Length": 0.355, "Rings": 7},
    {"Whole_weight": 0.451, "Length": 0.44, "Rings": 10},
    {"Whole_weight": 0.2555, "Length": 0.365, "Rings": 7},
    {"Whole_weight": 0.381, "Length": 0.45, "Rings": 9},
    {"Whole_weight": 0.2455, "Length": 0.355, "Rings": 11},
    {"Whole_weight": 0.2255, "Length": 0.38, "Rings": 10},
    {"Whole_weight": 0.9395, "Length": 0.565, "Rings": 12},
    {"Whole_weight": 0.7635, "Length": 0.55, "Rings": 9},
    {"Whole_weight": 1.1615, "Length": 0.615, "Rings": 10},
    {"Whole_weight": 0.9285, "Length": 0.56, "Rings": 11},
    {"Whole_weight": 0.9955, "Length": 0.58, "Rings": 11},
    {"Whole_weight": 0.931, "Length": 0.59, "Rings": 12},
    {"Whole_weight": 0.9365, "Length": 0.605, "Rings": 15},
    {"Whole_weight": 0.8635, "Length": 0.575, "Rings": 11},
    {"Whole_weight": 0.9975, "Length": 0.58, "Rings": 10},
    {"Whole_weight": 1.639, "Length": 0.68, "Rings": 15},
    {"Whole_weight": 1.338, "Length": 0.665, "Rings": 18},
    {"Whole_weight": 1.798, "Length": 0.68, "Rings": 19},
    {"Whole_weight": 1.7095, "Length": 0.705, "Rings": 13},
    {"Whole_weight": 0.4795, "Length": 0.465, "Rings": 8},
    {"Whole_weight": 1.217, "Length": 0.54, "Rings": 16},
    {"Whole_weight": 0.5225, "Length": 0.45, "Rings": 8},
    {"Whole_weight": 0.883, "Length": 0.575, "Rings": 11},
    {"Whole_weight": 0.3275, "Length": 0.355, "Rings": 9},
]


#Instantiation
backward_classifier = BackwardChainingClassifier()
accuracy, precision, recall, f1, cm = backward_classifier.evaluate(data)

print("Backward Chaining Results:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)