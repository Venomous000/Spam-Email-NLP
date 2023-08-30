# Spam-Email-NLP

# Text Classification using Neural Networks

This repository contains code for text classification using neural networks. The code uses the Keras library for building and training neural network models.

## Prerequisites

Before running the code, make sure you have the following libraries installed:
- `nltk`
- `pandas`
- `numpy`
- `keras`
- `sklearn`
- `matplotlib`

You can install these libraries using the following command:
pip install nltk pandas numpy keras scikit-learn matplotlib

## Dataset
The code uses a dataset named `spam.csv` for text classification. The dataset contains text messages labeled as either spam or non-spam (ham).

## Code Structure
- **Data Loading and Preprocessing**: The code reads the dataset using Pandas and prepares the input features `x` and target labels `y`. It also tokenizes the text data and performs one-hot encoding on the text using the Keras Tokenizer.
- **Model Definition and Training**: The code defines a simple neural network model using Keras. The initial model uses an embedding layer followed by a flatten layer and a dense layer. It is trained using the RMSprop optimizer and categorical cross-entropy loss. Training history is recorded.
- **Visualization**: The code visualizes the training and validation accuracy, as well as the training and validation loss using Matplotlib.
- **Improving the Model**: The code then replaces the initial model with an LSTM-based model to potentially improve performance. The LSTM model includes an embedding layer, two LSTM layers, and a dense layer. It is trained using the RMSprop optimizer and binary cross-entropy loss. Training history is again recorded and visualized.

## Usage
1. Clone this repository to your local machine.
git clone https://github.com/yourusername/text-classification-neural-network.git
cd text-classification-neural-network
2. Place the `spam.csv` dataset in the same directory as the code files.
3. Run the code using a Python interpreter.
python your_code_file.py
Replace `your_code_file.py` with the actual name of the Python file containing the code.
4. The code will execute and train the neural network models. Training progress and visualization will be displayed in the console and as plots.
