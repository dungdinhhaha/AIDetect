A. Chuẩn bị dữ liệu
Dữ liệu được lưu dưới dạng TFRecord (ảnh và nhãn).
File: read_tfrecord.py dùng để đọc dữ liệu vào pipeline.
B. Cấu hình
File: configs/config.py chứa các tham số như số class, đường dẫn dữ liệu, anchor, learning rate, v.v.
Bạn có thể chỉnh sửa các tham số này để phù hợp với bài toán.
C. Xây dựng mô hình
Backbone: ResNet-v2 (hoặc các mạng khác) để trích xuất đặc trưng từ ảnh.
FPN: Tạo feature pyramid để phát hiện vật thể ở nhiều kích thước (build_fpn.py).
RPN: Đề xuất các vùng có khả năng chứa vật thể (build_rpn.py).
Fast R-CNN: Phân loại và tinh chỉnh bounding box, sử dụng phương pháp Comparison Detector (build_fast_rcnn.py).
D. Training
File: train.py hoặc notebook train.ipynb dùng để huấn luyện mô hình.
Quá trình train gồm:
Load batch ảnh và nhãn.
Trích xuất đặc trưng qua backbone + FPN.
RPN sinh ra các proposals.
So sánh đặc trưng proposals với prototype từng class (Comparison Detector).
Tính loss (classification + regression).
Cập nhật trọng số mạng bằng optimizer (SGD).
E. Đánh giá và dự đoán
File: test.py hoặc notebook predict.ipynb dùng để dự đoán trên ảnh mới.
Flow dự đoán:
Load mô hình đã train.
Đọc ảnh đầu vào.
Chạy qua backbone, FPN, RPN, Fast R-CNN.
Hiển thị kết quả: bounding box, class, score (sử dụng visualize.py).
2️⃣ Chi tiết hoạt động từng bước
A. Training
Chạy lệnh:
hoặc mở train.ipynb và chạy từng cell.
Quá trình train sẽ lưu checkpoint mô hình vào thư mục cấu hình (MODLE_DIR).
B. Dự đoán
Chạy lệnh:
hoặc mở predict.ipynb và chạy từng cell.
Kết quả sẽ hiển thị ảnh với bounding box và nhãn dự đoán.
C. Các file quan trọng
label_dict.py: Định nghĩa mapping giữa tên class và số thứ tự.
libs/network_factory.py: Chọn backbone mạng.
visualize.py: Hiển thị kết quả dự đoán.
train.py, test.py: Script train và test.
3️⃣ Cách train dự án
Bước 1: Chuẩn bị môi trường
Cài đặt Python 3.6, TensorFlow 1.8.0, numpy, opencv, v.v.
Cài đặt các package cần thiết:
Bước 2: Chuẩn bị dữ liệu
Chuyển dữ liệu về dạng TFRecord (dùng convert_data_to_tfrecord.py nếu cần).
Đảm bảo đường dẫn dữ liệu đúng trong configs/config.py.
Bước 3: Train
Chạy script train:
hoặc mở notebook train.ipynb để train từng bước.
Bước 4: Đánh giá
Sau khi train xong, chạy test:
hoặc notebook predict.ipynb để dự đoán trên ảnh mới.
4️⃣ Tóm tắt luồng hoạt động
Đọc dữ liệu → Trích xuất đặc trưng → Sinh proposals → So sánh với prototype → Phân loại & điều chỉnh box → Tính loss → Cập nhật trọng số
Dự đoán: Load mô hình → Đọc ảnh → Chạy qua pipeline → Hiển thị kết quả