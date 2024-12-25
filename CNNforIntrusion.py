import pandas as pd
import numpy as np
import boto3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Mengunduh dataset dari S3
bucket_name = 'cnnforsecure'
file_name = 'UNSW-NB15.csv'

s3 = boto3.client('s3')
s3.download_file(bucket_name, file_name, 'UNSW-NB15.csv')

# Memuat dataset
data = pd.read_csv('UNSW-NB15.csv')

# Menampilkan head dan deskripsi dataset
print("Head of the dataset:")
print(data.head())
print("\nDescription of the dataset:")
print(data.describe())

# Pra-pemrosesan data
# Menghapus kolom yang tidak perlu (ganti dengan kolom yang sesuai)
data = data.drop(columns=['id', 'attack_cat'])  # Ganti dengan kolom yang sesuai jika ada

# Mengkodekan label
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])  # Ganti 'label' dengan nama kolom label yang sesuai

# Memisahkan fitur dan target
X = data.drop(columns=['label']).values  # Fitur
y = data['label'].values  # Target

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mengubah bentuk data untuk CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Definisikan fungsi aktivasi
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stabilitas numerik
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Definisikan kelas untuk CNN 1D
class CNN1D:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Inisialisasi bobot untuk lapisan konvolusi
        self.filters = np.random.randn(32, 3) * 0.01  # 32 filter, ukuran kernel 3
        self.bias = np.zeros((32, 1))
        
        # Inisialisasi bobot untuk lapisan dense
        self.W1 = np.random.randn(32 * (input_shape[0] - 2), 64) * 0.01  # 64 neuron
        self.b1 = np.zeros((64, 1))
        self.W2 = np.random.randn(64, num_classes) * 0.01  # Output layer
        self.b2 = np.zeros((num_classes, 1))

    def forward(self, X):
        # Forward pass
        self.X = X
        self.conv_out = self.convolution(X)
        self.pool_out = self.max_pooling(self.conv_out)
        self.flattened = self.pool_out.flatten().reshape(-1, 32 * (self.input_shape[0] - 2))
        self.hidden = relu(np.dot(self.flattened, self.W1) + self.b1)
        self.output = softmax(np.dot(self.hidden, self.W2) + self.b2)
        return self.output

    def convolution(self, X):
        # Operasi konvolusi
        batch_size = X.shape[0]
        conv_out = np.zeros((batch_size, 32, X.shape[1] - 2))  # 32 filter
        for i in range(batch_size):
            for j in range(32):
                conv_out[i, j] = np.convolve(X[i, :, 0], self.filters[j], mode='valid') + self.bias[j]
        return conv_out

    def max_pooling(self, conv_out):
        # Operasi max pooling
        batch_size = conv_out.shape[0]
        pooled_out = np.zeros((batch_size, 32, (conv_out.shape[2] // 2)))
        for i in range(batch_size):
            for j in range(32):
                pooled_out[i, j] = conv_out[i, j, ::2]  # Ambil setiap elemen kedua
        return pooled_out

    def backward(self, X, y, learning_rate=0.01):
        # Backward pass
        m = y.shape[0]
        output_loss = self.output - y
        dW2 = np.dot(self.hidden.T, output_loss) / m
        db2 = np.sum(output_loss, axis=0, keepdims=True) / m
        
        hidden_loss = np.dot(output_loss, self.W2.T) * relu_derivative(self.hidden)
        dW1 = np.dot(self.flattened.T, hidden_loss) / m
        db1 = np.sum(hidden_loss, axis=0, keepdims=True) / m
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=10, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            print(f'Epoch {epoch + 1}/{epochs} completed.')

# Menginisialisasi dan melatih model
model = CNN1D(input_shape=(X_train.shape[1], 1), num_classes=len(label_encoder.classes_))
model.train(X_train, y_train, epochs=10, batch_size=32)

# Evaluasi model
predictions = model.forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print(f'Test accuracy: {accuracy}')
