import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from joblib import load, dump
import mlflow
import mlflow.sklearn
import os
from mlflow.models.signature import infer_signature
import numpy as np


# Đọc dữ liệu đã xử lý
X_train = pd.read_csv("processed_data/X_train.csv")
X_valid = pd.read_csv("processed_data/X_valid.csv")
X_test = pd.read_csv("processed_data/X_test.csv")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_valid = pd.read_csv("processed_data/y_valid.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

# Cấu hình MLflow
mlflow.set_experiment("RandomForest_Classification")

# Khởi tạo mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

def main():
    st.set_page_config(layout="wide")  
    st.title("📊 Ứng dụng Phân Tích Dữ Liệu & Dự Đoán")

    tab1, tab2, tab3 = st.tabs(["📂 Xử lý dữ liệu", "📈 Huấn luyện mô hình", "🤖 Dự đoán"])

    # ========================== TAB 1: XỬ LÝ DỮ LIỆU ==========================
    with tab1:
        st.header("🔄 Quá trình Xử lý Dữ liệu")

        # Tải dữ liệu Titanic từ URL
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)

        # **1️⃣ Hiển thị Dữ liệu Gốc**
        st.subheader("📋 Dữ liệu gốc")
        st.write(df.head())

        # **2️⃣ Thông tin về dữ liệu**
        st.subheader("📊 Thông tin dữ liệu")
        st.write(f"🔹 **Số dòng và cột**: {df.shape}")
        st.write(f"🔹 **Các cột**: {df.columns}")
        st.write(f"🔹 **Dữ liệu thiếu** (NaN):")
        missing_values = df.isna().sum()
        st.write(missing_values[missing_values > 0])

        # **3️⃣ Hiển thị từng bước xử lý dữ liệu**
        st.subheader("🛠️ Các bước xử lý dữ liệu")

        with st.expander("📌 Bước 1: Loại bỏ cột không cần thiết"):
            drop_cols = ['Name', 'Ticket', 'Cabin']  # Các cột không cần thiết
            df = df.drop(columns=drop_cols)
            st.write(f"🔹 Cột đã loại bỏ: {drop_cols}")
            st.write(df.head())

        with st.expander("📌 Bước 2: Xử lý dữ liệu thiếu"):
            # Hiển thị số lượng dữ liệu thiếu trước khi xử lý
            st.write(f"🔹 Dữ liệu thiếu trước khi xử lý: {df.isna().sum()}")
            
            # Cụ thể xử lý các cột có dữ liệu thiếu
            st.write("🔹 **Xử lý cột 'Age'**: Điền giá trị thiếu bằng giá trị trung bình (median).")
            df['Age'] = df['Age'].fillna(df['Age'].median())
            
            st.write("🔹 **Xử lý cột 'Embarked'**: Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất (mode).")
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

            # Hiển thị số lượng dữ liệu thiếu sau khi xử lý
            st.write(f"🔹 Dữ liệu thiếu sau khi xử lý: {df.isna().sum()}")
            st.write(df.head())

        with st.expander("📌 Bước 3: Chuyển đổi One-Hot Encoding và Mã hóa"):
            # Mã hóa "Sex" theo cách mong muốn
            df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

            # Mã hóa "Embarked" theo cách mong muốn
            df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
            
            # Log các tham số vào MLflow
            mlflow.log_param("encoded_columns", ['Sex', 'Embarked'])

            # Hiển thị kết quả sau khi mã hóa
            st.write(f"🔹 Sau khi mã hóa: ")
            st.write(df.head())


        with st.expander("📌 Bước 4: Chuẩn hóa dữ liệu"):
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            numerical_cols = ['Age', 'Fare']
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            st.write(f"🔹 Dữ liệu sau khi chuẩn hóa: ")
            st.write(df.head())

        # **4️⃣ Hiển thị Dữ liệu Sau Xử Lý**
        st.subheader("✅ Dữ liệu sau xử lý")
        st.write(df.head())

        # **5️⃣ Chia tập dữ liệu**
        from Buoi2_processing import split_data
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df, target_column="Survived")

        # Hiển thị kích thước tập train/val/test
        st.subheader("📊 Kích thước tập dữ liệu sau khi chia")
        st.write(f"🔹 **Train:** {X_train.shape} mẫu")
        st.write(f"🔹 **Validation:** {X_valid.shape} mẫu")
        st.write(f"🔹 **Test:** {X_test.shape} mẫu")

        # Cho phép tải về dữ liệu sau xử lý
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Tải xuống dữ liệu đã xử lý", csv, "processed_data.csv", "text/csv")

    # ========================== TAB 2: HUẤN LUYỆN MÔ HÌNH ==========================
    with tab2:
            st.header("📈 Huấn luyện mô hình")

            # Hiển thị kích thước các tập dữ liệu
            st.subheader("📊 Kích thước các tập dữ liệu")
            st.write(f"🔹 **Train**: {X_train.shape[0]} mẫu")
            st.write(f"🔹 **Validation**: {X_valid.shape[0]} mẫu")
            st.write(f"🔹 **Test**: {X_test.shape[0]} mẫu")

            # Cross Validation Results
            st.subheader("📉 Kết quả Cross Validation")
            for i, acc in enumerate(cross_val_results):
                st.write(f"🔹 **Fold {i+1} Accuracy**: {acc:.4f}")

            # Kết quả trung bình của Cross Validation
            mean_cv_accuracy = cross_val_results.mean()
            st.write(f"🔹 **Kết quả trung bình Cross Validation**: {mean_cv_accuracy:.4f}")

            # Huấn luyện mô hình và dự đoán trên các tập dữ liệu
            rf_model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

            # Dự đoán trên các tập Train, Validation và Test
            y_train_pred = rf_model.predict(X_train)
            y_valid_pred = rf_model.predict(X_valid)
            y_test_pred = rf_model.predict(X_test)

            # Tính accuracy cho các tập Train, Validation, Test
            train_acc = accuracy_score(y_train, y_train_pred)
            valid_acc = accuracy_score(y_valid, y_valid_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Hiển thị kết quả accuracy
            st.subheader("📈 Kết quả accuracy")
            st.write(f"🔹 **Train Accuracy**: {train_acc:.4f}")
            st.write(f"🔹 **Validation Accuracy**: {valid_acc:.4f}")
            st.write(f"🔹 **Test Accuracy**: {test_acc:.4f}")

            # Hiển thị Classification Report theo dạng bảng đẹp hơn
            st.subheader("📑 Classification Report")

            def format_classification_report(report):
                """ Chuyển đổi classification_report thành DataFrame đẹp hơn """
                df = pd.DataFrame(report).T
                df = df.round(4)  # Làm tròn số thập phân để dễ đọc
                df.index.name = "Class"  # Đặt tên cho index
                return df

            # Hiển thị báo cáo cho từng tập dữ liệu
            st.write("🔹 **Train Set:**")
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            st.dataframe(format_classification_report(train_report))

            st.write("🔹 **Validation Set:**")
            valid_report = classification_report(y_valid, y_valid_pred, output_dict=True)
            st.dataframe(format_classification_report(valid_report))

            st.write("🔹 **Test Set:**")
            test_report = classification_report(y_test, y_test_pred, output_dict=True)
            st.dataframe(format_classification_report(test_report))


    # Kiểm tra và kết thúc run đang mở trước khi bắt đầu mới
    if mlflow.active_run():
        mlflow.end_run()

    # Bắt đầu ghi log vào MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)

        # Log kết quả Cross Validation vào MLflow
        for i, acc in enumerate(cross_val_results):
            mlflow.log_metric(f"fold_{i+1}_accuracy", acc)

        # Log kết quả accuracy vào MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("valid_accuracy", valid_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        # Log classification report vào MLflow
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        for label, metrics in class_report.items():
            if isinstance(metrics, dict):  # Chỉ log các giá trị dạng số
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"class_{label}_{metric_name}", value)

        # Kiểm tra và tạo thư mục lưu mô hình nếu chưa có
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "random_forest.pkl")

        # Lưu mô hình
        dump(rf_model, model_path)

        # Tạo input_example và signature để log model đúng cách
        input_example = X_test.iloc[:5]  # Lấy 5 dòng làm ví dụ đầu vào
        signature = infer_signature(X_test, y_test_pred)  # Tự động tạo signature từ dữ liệu test

    # Kết thúc phiên chạy MLflow
    mlflow.end_run()

# ========================== TAB 3: DỰ ĐOÁN ==========================
    with tab3:
        st.header("🚢 Titanic Survival Prediction")
        st.subheader("💡 Model Performance")

        # Load model
        model = load("models/random_forest.pkl")

        # Hiển thị form nhập liệu ngay trong tab, không dùng sidebar
        st.subheader("🔢 Nhập Thông Tin Hành Khách")

        col1, col2 = st.columns(2)  # Chia giao diện thành 2 cột để gọn gàng

        with col1:
            pclass = st.selectbox("🔹 Pclass", [1, 2, 3])
            sex = st.selectbox("🔹 Sex", ["Male", "Female"])
            age = st.number_input("🔹 Age", min_value=0, max_value=100, value=30)
            embarked = st.selectbox("🔹 Embarked", ["S", "C", "Q"])

        with col2:
            PassID = st.number_input("🔹 PassengerId", min_value=0, max_value=1000, value=1)
            sibsp = st.number_input("🔹 SibSp", min_value=0, max_value=10, value=0)
            parch = st.number_input("🔹 Parch", min_value=0, max_value=10, value=0)
            fare = st.number_input("🔹 Fare", min_value=0.0, max_value=500.0, value=32.0)

        # Mã hóa dữ liệu đầu vào
        sex = 0 if sex == "Male" else 1
        embarked = {"S": 0, "C": 1, "Q": 2}.get(embarked, -1)  # Mã hóa Embarked

        # Thêm các đặc trưng One-Hot Encoding nếu mô hình yêu cầu
        input_data = np.array([[PassID,pclass, sex, age, sibsp, parch, fare, embarked]], dtype=np.float64)

        # Nút dự đoán
        if st.button("🚀 Predict Survival"):
            try:
                prediction = model.predict(input_data)[0]
                result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
                st.subheader(f"🎯 Prediction: {result}")
            except Exception as e:
                st.error(f"🚨 Prediction Error: {e}")




if __name__ == "__main__":
    main()
