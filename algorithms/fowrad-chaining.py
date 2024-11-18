from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ForwardChainingClassifier:
    def __init__(self):
        # Rules for classification based on Whole_weight and Length
        self.rules = [
            {"condition": self.is_young, "label": "young"},
            {"condition": self.is_adult, "label": "adult"},
            {"condition": self.is_old, "label": "old"}
        ]

    def is_young(self, weight, length):
        return weight <= 0.4 and length <= 0.4

    def is_adult(self, weight, length):
        return 0.4 < weight <= 0.8 and 0.4 < length <= 0.6

    def is_old(self, weight, length):
        return weight > 0.8 or length > 0.6

    def predict(self, weight, length):
        for rule in self.rules:
            if rule["condition"](weight, length):
                return rule["label"]
        return "Unknown"

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
            prediction = self.predict(d["Whole_weight"], d["Length"])
            predictions.append(prediction)

        # Map labels to integers for evaluation (ensure unique labels from both true_labels and predictions)
        all_labels = list(set(true_labels + predictions))  # Union of true and predicted labels
        labels_map = {label: i for i, label in enumerate(all_labels)}
        
        # Convert true_labels and predictions into numeric labels
        y_true = [labels_map[label] for label in true_labels]
        y_pred = [labels_map[label] for label in predictions]

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        return accuracy, precision, recall, f1, cm

data = [
    {"Whole_weight": 0.514, "Length": 0.455, "Rings": 15},
    {"Whole_weight": 0.2255, "Length": 0.35, "Rings": 7},
    {"Whole_weight": 0.677, "Length": 0.53, "Rings": 9},
    {"Whole_weight": 0.516, "Length": 0.44, "Rings": 10},
    {"Whole_weight": 0.205, "Length": 0.33, "Rings": 7},
    {"Whole_weight": 0.8945, "Length": 0.55, "Rings": 19},
    {"Whole_weight": 1.1615, "Length": 0.615, "Rings": 10},
    {"Whole_weight": 0.2905, "Length": 0.355, "Rings": 7},
    {"Whole_weight": 0.7635, "Length": 0.55, "Rings": 9},
    {"Whole_weight": 0.381, "Length": 0.45, "Rings": 9},
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
