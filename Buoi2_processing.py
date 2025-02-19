import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(url):
    """Tải dữ liệu từ URL."""
    df = pd.read_csv(url)
    mlflow.log_param("data_source", url)
    mlflow.log_param("original_shape", df.shape)
    return df

def preprocess_data(df):
    """Xử lý dữ liệu Titanic."""
    
    # 1️⃣ Xóa cột không cần thiết
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=drop_cols)
    mlflow.log_param("dropped_columns", drop_cols)
    
    # 2️⃣ Xử lý dữ liệu thiếu
    missing_values_before = df.isnull().sum().to_dict()
    mlflow.log_dict(missing_values_before, "missing_values_before.json")
    
    df = df.assign(
        Age=df['Age'].fillna(df['Age'].median()),
        Embarked=df['Embarked'].fillna(df['Embarked'].mode()[0]),
        Fare=df['Fare'].fillna(df['Fare'].median())
    )
    mlflow.log_param("filled_missing_columns", ['Age', 'Embarked', 'Fare'])
    
    # 3️⃣ Chuyển đổi dữ liệu phân loại
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    mlflow.log_param("encoded_columns", ['Sex', 'Embarked'])
    
    # 4️⃣ Chuẩn hóa dữ liệu số
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Fare']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    mlflow.log_param("numerical_columns_standardized", numerical_cols)
    
    # Ghi log số lượng dữ liệu thiếu sau xử lý
    missing_values_after = df.isnull().sum().to_dict()
    mlflow.log_dict(missing_values_after, "missing_values_after.json")
    mlflow.log_param("processed_shape", df.shape)
    
    return df

def split_data(df, target_column='Survived'):
    """Chia dữ liệu thành tập train/valid/test (70/15/15)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("valid_size", len(X_valid))
    mlflow.log_metric("test_size", len(X_test))
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def save_processed_data(X_train, X_valid, X_test, y_train, y_valid, y_test, output_dir="processed_data"):
    """Lưu dữ liệu đã xử lý."""
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_valid.to_csv(os.path.join(output_dir, 'X_valid.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_valid.to_csv(os.path.join(output_dir, 'y_valid.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    mlflow.log_artifacts(output_dir)
    print(f"✅ Dữ liệu đã được xử lý và lưu thành công tại: {output_dir}")

if __name__ == "__main__":
    mlflow.set_experiment("Titanic_Data_Processing")
    
    with mlflow.start_run():
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = load_data(url)
        df = preprocess_data(df)
        X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
        save_processed_data(X_train, X_valid, X_test, y_train, y_valid, y_test)
        print("✅ Dữ liệu đã được xử lý và lưu thành công!")
    
    mlflow.end_run()
