import pandas as pd
# Đọc dữ liệu từ file csv vào DataFrame
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)
# Đặt tên cho các cột
df.columns = ['top_left', 'top_middle', 'top_right', 'middle_left', 'middle_middle', 'middle_right', 'bottom_left', 'bottom_middle', 'bottom_right', 'class']
import numpy as np

# Đổi tên giá trị trong cột "class": negative thành 0 và positive thành 1
df['class'] = np.where(df['class'] == 'positive', 1, 0)

# Sử dụng get_dummies để chuyển các giá trị không phải số thành dạng số.
df = pd.get_dummies(df)
# Tách dữ liệu thành tập huấn luyện và tập kiểm tra, tỉ lệ 80:20.
train = df.sample(frac=0.8, random_state=123)
test = df.drop(train.index)
from sklearn.tree import DecisionTreeClassifier

# Tạo mô hình Decision Tree
model = DecisionTreeClassifier(max_depth=3)

# Huấn luyện mô hình
model.fit(train.iloc[:, :-1], train.iloc[:, -1])
from sklearn.metrics import accuracy_score

# Dự đoán trên tập kiểm tra
y_pred = model.predict(test.iloc[:, :-1])

# Tính độ chính xác của mô hình
accuracy = accuracy_score(test.iloc[:, -1], y_pred)

# In ra độ chính xác của mô hình
print("Độ chính xác của mô hình Decision Tree là:", accuracy)
