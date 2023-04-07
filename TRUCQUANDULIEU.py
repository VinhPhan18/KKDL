##1.2. Trực qua hóa tập dữ liệuimport pandas as pd

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('tic-tac-toe.csv', header=None)

# # Đặt tên cho các cột
# df.columns = ['pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7', 'pos8', 'pos9', 'result']

# # Hiển thị bảng dữ liệu (với 5 dòng đầu tiên)
# print(df.head())

# import pandas as pd

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('tic-tac-toe.csv', header=None)

# # Đặt tên cho các cột
# df.columns = ['pos1', 'pos2', 'pos3', 'pos4', 'pos5', 'pos6', 'pos7', 'pos8', 'pos9', 'result']

# # Kiểm tra các giá trị null trong tập dữ liệu
# print(df.isnull())




#####cot du lieu top-left-square
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Định nghĩa tên cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ biểu đồ countplot cho cột "top-left-square"
# sns.countplot(x='top-left-square', data=data)
# plt.show()




# ##cot du lieu top-middle-square
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ biểu đồ countplot cho cột "top-middle-square"
# sns.countplot(x='top-middle-square', data=data)

# # Hiển thị biểu đồ
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đổi tên các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ plot count cho cột "top-right-square"
# sns.countplot(x='top-right-square', data=data)

# # Hiển thị plot
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đổi tên các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ plot count cho cột "middle-left-square"
# sns.countplot(x='middle-left-square', data=data)

# # Hiển thị plot
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 
#                 'middle-left-square', 'middle-middle-square', 'middle-right-square', 
#                 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ biểu đồ countplot cho cột "middle-middle-square"
# sns.countplot(x='middle-middle-square', data=data)

# # Hiển thị biểu đồ
# plt.show()



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 
#                 'middle-left-square', 'middle-middle-square', 'middle-right-square', 
#                 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ biểu đồ countplot cho cột "middle-right-square"
# sns.countplot(x='middle-right-square', data=data)

# # Hiển thị biểu đồ
# plt.show()



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đổi tên các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ plot count cho cột "bottom-left-square"
# sns.countplot(x='bottom-left-square', data=data)

# # Hiển thị plot
# plt.show()



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# # Đọc dữ liệu từ file csv
# data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# data.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 
#                 'middle-left-square', 'middle-middle-square', 'middle-right-square', 
#                 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Biểu đồ countplot cho cột "bottom-middle-square"
# sns.countplot(x="bottom-middle-square", data=data)

# # Tiêu đề cho biểu đồ
# plt.title("Biểu đồ số lượng trường hợp có giá trị khác nhau trong cột 'bottom-middle-square'")

# # Hiển thị biểu đồ
# plt.show()



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data',header=None)

# # Đặt tên cho các cột
# df.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 
#                 'middle-left-square', 'middle-middle-square', 'middle-right-square', 
#                 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Biểu đồ countplot cho cột "bottom-right-square"
# sns.countplot(x='bottom-right-square', data=df)

# # Đặt tiêu đề cho biểu đồ
# plt.title('Số lượng trường hợp của các giá trị khác nhau trong cột  "bottom-right-square"')

# # Hiển thị biểu đồ
# plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đổi tên các cột
# df.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 
#               'middle-left-square', 'middle-middle-square', 'middle-right-square', 
#               'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Vẽ biểu đồ countplot cho cột "class"
# sns.countplot(x='class', data=df)

# # Thiết lập tiêu đề cho biểu đồ
# plt.title('Biểu đồ số lượng trường hợp của các giá trị của cột "class"')

# # Hiển thị biểu đồ
# plt.show()



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đổi tên các cột
# df.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 
#               'middle-left-square', 'middle-middle-square', 'middle-right-square', 
#               'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']

# # Tính toán ma trận tương quan giữa các biến
# corr = df.corr()

# # Hiển thị ma trận tương quan trên console
# print(corr)

# # Vẽ biểu đồ heatmap cho tập dữ liệu
# sns.heatmap(corr, annot=True, cmap='coolwarm')

# # Thiết lập tiêu đề cho biểu đồ
# plt.title('Biểu đồ heatmap cho tập dữ liệu Tic-Tac-Toe Endgame')

# # Hiển thị biểu đồ
# plt.show()


# ###Mối tương quan giữa các biếnb
####vẽ biểu đồ tần suất cho mỗi vị trí trên bàn cờ
# import pandas as pd
# import matplotlib.pyplot as plt

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# df.columns = ['top_left', 'top_middle', 'top_right', 'middle_left', 'middle_middle', 'middle_right', 'bottom_left', 'bottom_middle', 'bottom_right', 'class']

# # Tính tần suất cho mỗi vị trí
# frequency = df.apply(pd.Series.value_counts)
# del frequency['class']

# # Vẽ biểu đồ tần suất
# ax = frequency.plot(kind='bar', figsize=(10,6), fontsize=12)
# ax.set_xlabel('Vị trí trên bàn cờ', fontsize=12)
# ax.set_ylabel('Số lượng', fontsize=12)
# ax.set_title('Tần suất của các vị trí trên bàn cờ Tic-Tac-Toe', fontsize=14)
# plt.show()




# ##sử dụng phương pháp Decision Tree để dự đoán kết quả trận đấu Tic-Tac-Toe Endgame
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import numpy as np

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# df.columns = ['top_left', 'top_middle', 'top_right', 'middle_left', 'middle_middle', 'middle_right', 'bottom_left', 'bottom_middle', 'bottom_right', 'class']

# # Chuyển các giá trị không số thành số
# df['class'] = np.where(df['class'] == 'positive', 1, 0)
# df = pd.get_dummies(df)

# # Tách tập huấn luyện và tập kiểm tra
# train = df.sample(frac=0.8, random_state=123)
# test = df.drop(train.index)

# # Tạo mô hình Decision Tree
# model = DecisionTreeClassifier(max_depth=3)

# # Huấn luyện mô hình
# model.fit(train.iloc[:, :-1], train.iloc[:, -1])

# # Dự đoán trên tập kiểm tra
# y_pred = model.predict(test.iloc[:, :-1])

# # Tính độ chính xác của mô hình
# accuracy = accuracy_score(test.iloc[:, -1], y_pred)

# # In ra độ chính xác của mô hình
# print("Độ chính xác của mô hình Decision Tree là:", accuracy)




###mo hinh RandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import numpy as np

# # Đọc dữ liệu từ file csv
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', header=None)

# # Đặt tên cho các cột
# df.columns = ['top_left', 'top_middle', 'top_right', 'middle_left', 'middle_middle', 'middle_right', 'bottom_left', 'bottom_middle', 'bottom_right', 'class']

# # Chuyển các giá trị không số thành số
# df['class'] = np.where(df['class'] == 'positive', 1, 0)
# df = pd.get_dummies(df)

# # Tách tập huấn luyện và tập kiểm tra
# train = df.sample(frac=0.8, random_state=123)
# test = df.drop(train.index)

# # Tạo mô hình Random Forest
# model = RandomForestClassifier(n_estimators=100, max_depth=3)

# # Huấn luyện mô hình
# model.fit(train.iloc[:, :-1], train.iloc[:, -1])

# # Dự đoán trên tập kiểm tra
# y_pred = model.predict(test.iloc[:, :-1])

# # Tính độ chính xác của mô hình
# accuracy = accuracy_score(test.iloc[:, -1], y_pred)

# # In ra độ chính xác của mô hình
# print("Độ chính xác của mô hình RandomForest:", accuracy)



