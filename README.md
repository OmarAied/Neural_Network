# MNIST Handwritten Digit Recognition Neural Network

A simple yet effective 3-layer neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset.

## 🚀 Features

- **Pure NumPy Implementation**: No deep learning frameworks - built from scratch
- **3-Layer Neural Network**: 784 → 128 → 64 → 10 architecture
- **He Initialization**: Proper weight initialization for ReLU activations
- **ReLU Activation**: Hidden layers use Rectified Linear Units
- **Softmax Output**: Multi-class classification with 10 digit classes
- **Modular Design**: Separate training and testing scripts

## 📁 Project Structure

```
mnist-neural-network/
├── train_model.py          # Script to train and save the model
├── test_model.py          # Script to test and visualize predictions
├── trained_model.pkl      # Saved model parameters (gitignored)
├── mnist_train.csv        # Training dataset (gitignored)
├── mnist_test.csv         # Test dataset (gitignored)
└── README.md              # This file
```

## 🛠️ Installation & Setup

1. **Clone the repository** (if applicable)
2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib
   ```
3. **Download MNIST datasets** and place them in the project directory:
   - `mnist_train.csv` (60,000 training samples)
   - `mnist_test.csv` (10,000 test samples)

## 🧠 Model Architecture

```
Input Layer:   784 neurons (28×28 pixels)
Hidden Layer 1: 128 neurons (ReLU activation)
Hidden Layer 2: 64 neurons (ReLU activation)  
Output Layer:   10 neurons (Softmax activation - digits 0-9)
```

**Key Techniques:**
- **He Initialization**: `w = random * sqrt(2.0 / fan_in)`
- **Batch Gradient Descent**: Full dataset updates
- **Learning Rate**: 0.1
- **Training Iterations**: 500

## 📊 Usage

### 1. Train the Model
```bash
python train_model.py
```
This will:
- Load and preprocess the training data
- Train the neural network for 500 iterations
- Evaluate on a development set (1,000 samples)
- Save the trained parameters to `trained_model.pkl`

### 2. Test the Model
```bash
python test_model.py
```
This will:
- Load the saved model
- Test on a random sample from the test set
- Display the image with true vs predicted label

### 3. Manual Testing
```python
# In your own script
from test_model import test_prediction

# Test specific indices
test_prediction(index=42)           # Test sample 42
test_prediction(index=100)          # Test sample 100

# Or get all predictions
from test_model import make_prediction
predictions = make_prediction()
```

## 📈 Performance

- **Training Accuracy**: ~95.5% 
- **Development Set Accuracy**: ~94.4%
- **Test Set Accuracy**: ~94.0%
- **Training Time**: ~5-10 minutes (depending on hardware)

## 📚 Technical Details

### Forward Propagation
```python
Z1 = w1·X + b1
A1 = ReLU(Z1)
Z2 = w2·A1 + b2  
A2 = ReLU(Z2)
Z3 = w3·A2 + b3
A3 = softmax(Z3)  # Output probabilities
```

### Backward Propagation
- Cross-entropy loss with softmax
- Chain rule for gradient computation
- Batch gradient updates

### Data Preprocessing
- Pixel values normalized to [0, 1] range
- Labels one-hot encoded for training
- Random shuffling of training data

## 🎯 Future Improvements

Potential enhancements to consider:
- Add dropout regularization
- Implement learning rate decay
- Add batch normalization
- Convert to mini-batch gradient descent
- Add early stopping
- Implement L2 regularization
- Create a GUI for drawing and predicting digits


*Built with Python, NumPy, and lots of linear algebra!*
