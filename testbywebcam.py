import atexit
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'TrafficSign_model.h5'
model = load_model(MODEL_PATH)

# Đặt danh sách các file tạm thời bạn muốn xóa
temp_files_to_delete = ['temp_image.jpg']

# Lấy đường dẫn thư mục làm việc hiện tại
current_working_directory = os.getcwd()

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    # Các thông tin về tên lớp, bạn có thể điều chỉnh dựa trên mô hình và dữ liệu của mình
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vechiles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vechiles', 'Vechiles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vechiles over 3.5 metric tons'
    ]
    return class_names[classNo]

def model_predict_with_prob(img):
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    classProb = predictions[0, classIndex]
    preds = getClassName(classIndex)
    return preds, classProb

def cleanup_temp_files():
    for temp_file in temp_files_to_delete:
        full_temp_file_path = os.path.join(current_working_directory, temp_file)
        try:
            os.remove(full_temp_file_path)
            print(f"Deleted {full_temp_file_path}")
        except FileNotFoundError:
            pass  # Nếu file không tồn tại, bỏ qua

# Đăng ký hàm cleanup_temp_files() để được gọi khi chương trình kết thúc
atexit.register(cleanup_temp_files)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Dự đoán từ frame hiện tại
        preds, classProb = model_predict_with_prob(frame)

        # Hiển thị kết quả dự đoán và xác suất trực tiếp trên frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{preds} ({classProb * 100:.2f}%)"
        cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Hiển thị frame
        cv2.imshow('Traffic Sign Detection', frame)

        # Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()