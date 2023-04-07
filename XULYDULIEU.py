import pandas as pd

# Đọc dữ liệu từ file csv vào DataFrame
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)
# Đặt tên cho các cột
df.columns = ['top_left', 'top_middle', 'top_right', 'middle_left', 'middle_middle',
               'middle_right', 'bottom_left', 'bottom_middle', 'bottom_right', 'class']
import numpy as np

# Đổi tên giá trị trong cột "class": negative thành 0 và positive thành 1
df['class'] = np.where(df['class'] == 'positive', 1, 0)

# Sử dụng get_dummies để chuyển các giá trị không phải số thành dạng số.
df = pd.get_dummies(df)
# Tách dữ liệu thành tập huấn luyện và tập kiểm tra, tỉ lệ 80:20.
train = df.sample(frac=0.8, random_state=123)
test = df.drop(train.index)
# Chọn các biến đầu vào và biến đầu ra
X_train = train.drop('class', axis=1)
y_train = train['class']

X_test = test.drop('class', axis=1)
y_test = test['class']
