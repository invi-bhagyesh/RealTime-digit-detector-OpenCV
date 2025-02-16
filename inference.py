import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTCNN
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_path, batch_size=512):

    device = torch.device("cpu")
    
    # Load model
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Collect predictions and labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1) # we dont need max value, only index
            
            all_preds.extend(preds.cpu().numpy()) # convert to numpy array
            all_labels.extend(labels.numpy()) # add current batch prediction to all_pred list]
    
    # Generate evaluation metrics
    print("Test Set Evaluation:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Calculate final accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model("mnist_cnn.pth")

    '''
final accuracy => 99.45


Confusion Matrix:
[[ 978    0    1    0    0    0    0    1    0    0]
 [   0 1133    1    0    0    0    0    1    0    0]
 [   0    1 1030    0    0    0    0    1    0    0]
 [   0    0    0 1006    0    1    0    2    1    0]
 [   0    0    0    0  978    0    0    0    0    4]
 [   1    0    0    3    0  886    1    1    0    0]
 [   2    1    0    0    1    2  948    0    4    0]
 [   0    2    5    0    0    0    0 1020    0    1]
 [   1    0    0    1    2    0    0    0  969    1]
 [   0    0    0    0    5    4    0    3    1  996]]


 Test Set Evaluation:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       980
           1       1.00      1.00      1.00      1135
           2       0.99      1.00      1.00      1032
           3       1.00      1.00      1.00      1010
           4       0.99      1.00      0.99       982
           5       0.99      0.99      0.99       892
           6       1.00      0.99      0.99       958
           7       0.99      0.99      0.99      1028
           8       0.99      0.99      0.99       974
           9       0.99      0.99      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
'''