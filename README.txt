------- Train model Convolutional Neural Networks để viết phần mềm nhận dạng biển báo giao thông -------

*** CÁCH CHẠY CHƯƠNG TRÌNH
Ứng dụng hỗ trợ: Google Colab, Visual Studio Code, Python, Jupyter Notebook,...
*** THƯ VIỆN CẦN THIẾT
OpenCV, Tensorflow, Keras, Numpy, Flask,...

*** XÂY DỰNG PHẦN MỀM ***

I. TRAIN MODEL (file main.py thực hiện việc đọc dataset từ thư mục Dataset(thư mục này chứa các tập hình ảnh biển báo giao thông)
và và thư mục lables.csv(chứa các nhãn biển báo), từ dataset đó tiến hành chuẩn hóa các hình ảnh đầu vào để tiến hành Train model để đánh giá được độ
chính xác từ dự đoán của Model đã được train trước đó và lưu thực hành bước lưu Model để dùng cho xây dựng phần mềm.)

-----> Mở file TrafficSignTrainModel.ipynb để xem lại nhật ký các bước thực hiện train model có trong file main.py

II. XÂY DỰNG APP TRÊN WEB(bước này thực hiện việc sử dụng Model đã lưu ở bước TRAIN MODEL trên và chúng ta sẽ tạo một trang web có chức 
năng là đưa vào một hình ảnh biển báo giao thông bất kì từ máy tính(cụ thể là các ảnh có sẵn trong thư mục upload) cho Model dự đoán 
được biển báo giao thông trong ảnh là biển báo gì)

-----> Mở chạy file TestOnWeb.py (file này sẽ tạo một trang web với đường dẫn có cổng là 5000, ở trang web này chúng ta thực hiện các chức
năng đã được mô tả ở trên)

III. XÂY DỰNG APP DỰ ĐOÁN BIỂN BÁO GIAO THÔNG REALTIME(bước này sẽ cho Model dự đoán biển báo có input truyền vào là 1 ảnh chứa biển báo
giao thông từ webcam, Model sẽ dự đoán ảnh đó là biển báo gì và kèm theo tỉ lệ dự đoán đúng của Model)
-----> Mở chạy file TestByWebcam.py(file này sẽ tạo một màn hình console tích hợp với webcam của laptop, từ webcam này có thể sử dụng một
số hình ảnh được chuẩn bị trong thư mục TrafficSignTest-IMG để mà truyền vào cho Model dự đoán)
Lưu ý: Nhấn nút q để kết thúc chương trình!

*** Xin cảm ơn! ***