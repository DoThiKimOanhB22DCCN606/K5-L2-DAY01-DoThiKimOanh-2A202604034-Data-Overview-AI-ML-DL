# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh chứ không phải về một vật thể riêng lẻ trong ảnh.
- Lý do để nhận biết:
+ Loại tác vụ: Trường "task": "image_classification" thể hiện rõ đây là tác vụ Phân loại ảnh.
+ Không có tọa độ vị trí: Record này hoàn toàn không chứa bất kỳ tọa độ hộp giới hạn dạng bbox_xyxy hay đa giác polygon_xy nào để chỉ ra vị trí của chiếc xe cab (taxi). Nó chỉ ghi nhận thông tin kích thước tổng thể của ảnh (image_width, image_height).
+ Cơ chế xếp hạng: Việc có "rank": 1 kết hợp với model phân loại yolo11n-cls.pt (ImageNet-1K) cho thấy mô hình đang đưa ra dự đoán khả thi nhất (Top-1) cho chủ đề bao quát của toàn bộ bức ảnh traffic này.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp không phải do model tự nghĩ ra, mà nó đi kèm với taxonomy của checkpoint mô hình đã được huấn luyện sẵn.
- Trong tệp classification_predictions.json, tất cả các bản ghi phân loại đều sử dụng chung một taxonomy_name là "ImageNet-1K" đi kèm với "model_file": "yolo11n-cls.pt". Danh sách 1000 lớp này (trong đó lớp cab có class_id: 468, lớp minibus có class_id: 654) đã được xác định cố định từ trước khi huấn luyện mô hình.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Để tránh trùng lặp và nhầm lẫn danh tính: Các bộ dữ liệu khác nhau có thể sử dụng cùng một tên lớp (class_name) nhưng lại đại diện cho các đối tượng khác nhau hoặc có ID khác nhau. Ghi đầy đủ class_id giúp định danh chính xác duy nhất lớp đó trong một hệ thống.
- Để xác định ngữ cảnh dữ liệu: Ghi taxonomy_name (ví dụ: ImageNet-1K hay COCO-80) giúp người dùng hoặc lập trình viên biết lớp này thuộc bộ quy chuẩn nào, tránh nhầm lẫn khi tích hợp hoặc so sánh nhiều mô hình với nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ nguyên tắc ưu tiên (Priority Rules). Trong tác vụ phân loại ảnh (chỉ được chọn 1 lớp duy nhất), guideline cần chỉ rõ:
+ Ưu tiên gán nhãn cho đối tượng chiếm diện tích lớn nhất (dominant object) hoặc nằm ở trung tâm bức ảnh.
+ Hoặc ưu tiên gán nhãn theo chủ đề cốt lõi của bài toán cần giải quyết (ví dụ: nếu làm ứng dụng giao thông, phải ưu tiên các lớp phương tiện giao thông như cab, bus hơn là các yếu tố nền cảnh).
+ Quy định rõ cách xử lý khi các đối tượng có tỉ lệ tương đương nhau để tránh việc annotator tự ý phán đoán vô căn cứ.
6. Vì sao không được lấy `score` của model làm ground truth?
+ score chỉ là độ tin cậy toán học của mô hình: Chỉ số score (ví dụ: 0.510915 cho lớp cab trong file JSON) chỉ thể hiện mức độ tự tin của thuật toán dựa trên những gì nó đã được học, hoàn toàn không phải thước đo mức độ chính xác tuyệt đối.
+ Mô hình có thể dự đoán sai: Mô hình có thể đạt score rất cao cho một dự đoán sai lệch (overconfidence) hoặc score rất thấp đối với một vật thể khó nhận biết nhưng thực tế vẫn tồn tại.
+ Ground Truth yêu cầu tính chính xác tuyệt đối từ con người: Nhãn chuẩn (Ground Truth) đại diện cho sự thật khách quan do con người (Annotator/Reviewer) xác minh dựa trên thế giới thực để làm thước đo chuẩn mực đánh giá và huấn luyện mô hình.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
class_name: "person"
score: 0.908585
bbox_xyxy: [384.81, 199.11, 492.71, 521.9]
bbox_width: 107.9
bbox_height: 322.79
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Bounding box này khoanh vùng đối tượng người đứng ở phía bên phải bức ảnh:
- Góc trên - trái: Tọa độ pixel là (384.81, 199.11) (cách lề trái ảnh 384.81 pixel và lề trên 199.11 pixel).
- Góc dưới - phải: Tọa độ pixel là (492.71, 521.9) (cách lề trái ảnh 492.71 pixel và lề trên 521.9 pixel).
- Kích thước: Hộp giới hạn này rộng 107.9 pixel và cao 322.79 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng 0.20: Có 17 predictions.
- Ngưỡng 0.60: Có 6 predictions.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
 Khi hạ ngưỡng tin cậy (nhận diện được nhiều hộp hơn), mô hình sẽ sinh ra rất nhiều hộp dự đoán có độ tin cậy thấp (thường bị trùng lặp, nhiễu hoặc sai lệch lớp). Người reviewer lúc này sẽ phải:
- Tốn nhiều công sức để rà soát và xóa bỏ các hộp dự đoán sai hoàn toàn (false positives).
- Chỉnh sửa lại các hộp bị lệch biên hoặc gộp các hộp bị trùng lặp trên cùng một đối tượng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Quy tắc bao bọc sát biên (Tight Bounding Box): "Người gán nhãn phải vẽ hộp giới hạn bao bọc sát nhất có thể vào các điểm ngoài cùng của vật thể (trên, dưới, trái, phải). Không được để thừa quá nhiều khoảng trống ngoài biên của vật thể, đồng thời tuyệt đối không cắt lẹm vào phần cơ thể hoặc cấu trúc chính của vật thể gán nhãn."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật thể bị che khuất/cắt mép: Chiếc thớt hoặc đĩa bánh nằm ở góc dưới-trái bàn bếp bị che khuất một phần bởi các bát trộn lớn, hoặc chiếc cốc bị khuất sau tay người đầu bếp.
- Hướng giải quyết: Đưa vào Guideline kèm ví dụ minh họa trực quan thay vì Escalation.
- Lý do: Hiện tượng vật thể bị che khuất (occlusion) hoặc bị cắt mép ảnh (truncation) xảy ra rất phổ biến trong các dự án thị giác máy tính thực tế. Việc quy định rõ ràng tỷ lệ che khuất (ví dụ: gán nhãn nếu nhìn thấy >20% vật thể) trong Guideline sẽ giúp hàng loạt annotator tự tin làm việc một cách nhất quán mà không cần tốn thời gian chờ đợi phản hồi từ Lab Coach cho từng bức ảnh đơn lẻ.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
instance_id: "kitchen-001"
class_name: "person"
score: 0.902621
polygon_point_count: 68
3 đến 5 điểm đầu của polygon_xy: [[432.0, 199.11], [431.0, 200.0], [428.0, 200.0], [424.0, 204.0], [423.0, 204.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Bounding Box (Hộp giới hạn): Chỉ cung cấp phạm vi bao quát ngoài cùng của vật thể dưới dạng hình chữ nhật thô sơ ([x_min, y_min, x_max, y_max]), chứa rất nhiều pixel nhiễu thuộc về nền (background) hoặc các vật thể khác đè lên.
- Polygon (Đa giác): Cung cấp thông tin hình học chính xác bám sát theo đường biên thực tế của vật thể. Nó phân tách rõ ràng pixel nào thuộc về đối tượng và pixel nào thuộc về nền, giúp mô hình nắm bắt được hình dáng thực tế (như tư thế của người đầu bếp, độ cong của chiếc bát).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- instance_id (ví dụ: kitchen-001) dùng để phân biệt và định danh từng cá thể độc lập trong cùng một bức ảnh.
- Hiểu câu nói "không phải class ID và không phải tracking ID":
+ Không phải class ID: Class ID đại diện cho phân loại chung (ví dụ lớp person có class_id là 0). Ở đây, nếu có 2 người, họ dùng chung class_id nhưng sẽ có hai instance_id khác nhau (ví dụ kitchen-001 và kitchen-006).
+ Không phải tracking ID: Tracking ID dùng để định danh duy nhất một đối tượng xuyên suốt qua nhiều khung hình khác nhau của một chuỗi video. Còn instance_id trong bài lab này chỉ có giá trị tĩnh, được đánh số thứ tự tạm thời cho các thực thể được phát hiện trong phạm vi duy nhất một bức ảnh tĩnh này.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc gán nhãn sát biên (Edge-Aligned Boundary Rule): "Khi vẽ đa giác, các điểm mốc (vertices) phải được đặt ngay trên đường ranh giới phân tách giữa vật thể và nền. Khoảng cách giữa hai điểm liên tiếp không được vượt quá 10 pixel ở các đường cong phức tạp để tránh làm biến dạng hình dáng thực tế của vật thể; không được cắt lẹm vào trong vật thể và không được lấn ra ngoài nền quá 2 pixel."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Hiện tượng trên ảnh kitchen: Có một khu vực rất phức tạp ở góc dưới bên trái, nơi chồng chéo và chạm nhau của nhiều chiếc bát (bowl) chồng lên nhau và đặt trên bàn ăn (dining table), cùng với chiếc gậy cán bột. Biên giới giữa các chiếc bát rất mờ và chồng lấn nghiêm trọng.
- Hướng giải quyết: Đưa vào Guideline chi tiết thay vì Escalation.
- Lý do: Các trường hợp vật thể chạm nhau hoặc chồng lấn (occlusion) là các tình huống kinh điển và xuất hiện liên tục trong gán nhãn công nghiệp. Nếu mỗi lần gặp ta đều escalate lên Lab Coach sẽ gây nghẽn tiến độ dự án (bottleneck). Thay vào đó, ta cần thiết lập quy tắc cứng trong Guideline: "Đối với các vật thể xếp chồng, vẽ trọn vẹn phần nhìn thấy được của từng vật thể dưới dạng một thực thể riêng biệt. Nếu phần bị che khuất nhỏ hơn 50%, annotator tự ước lượng và vẽ bù phần biên bị che khuất theo quán tính hình học của vật thể đó."

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Là một nhãn lớp duy nhất (Single-label class ID) gán cho toàn bộ ảnh từ danh mục cố định (ví dụ: `cab`, `minibus`). | Ảnh `traffic` chứa rất nhiều loại phương tiện đan xen, việc chỉ chọn một nhãn duy nhất như `cab` (score: 0.51) làm mất mát thông tin của các đối tượng khác như `bus` hay `car`. | Đọc kỹ guideline định nghĩa các lớp phân loại; quan sát toàn cảnh bức ảnh và chọn một nhãn phù hợp nhất đại diện cho nội dung chính của ảnh. | Kiểm tra xem nhãn được chọn có bị lệch lạc, chủ quan hoặc sai sót so với định nghĩa chuẩn của taxonomy hay không. |
| **Phát hiện vật thể** | Là một tập hợp các hộp giới hạn dưới dạng tọa độ pixel hình chữ nhật `[x_min, y_min, x_max, y_max]` kèm nhãn lớp cho từng vật thể độc lập. | Trong ảnh `kitchen`, mô hình nhận diện sai thớt gỗ và bột mì thành `bowl` (score 0.72/0.50) và bỏ sót các dụng cụ nấu ăn nhỏ như thìa/muỗng (`spoon`) treo trên tường ở ngưỡng lọc 0.35. | Vẽ các hộp giới hạn (Bounding box) ôm sát từng vật thể mục tiêu; đảm bảo không vẽ thiếu hoặc thừa ranh giới của vật thể; gán nhãn lớp chuẩn cho mỗi hộp. | Đối chiếu hộp giới hạn xem có bị cắt lẹm hoặc quá rộng không; kiểm tra các vật thể bị bỏ sót hoặc gán nhãn sai loại (như lỗi thớt biến thành tô). |
| **Instance segmentation** | Là một tập hợp điểm tọa độ tạo thành đa giác khép kín (Polygon `[[x1, y1], [x2, y2], ...]`) bám sát ranh giới của từng thực thể riêng biệt. | Vùng đa giác của chiếc bàn ăn (`dining table` score 0.58) bị lem nhem, chồng chéo không rõ ràng lên các khay bột mì; ranh giới các dụng cụ treo tường quá phức tạp và mờ để xác định chính xác pixel. | Sử dụng công cụ vẽ đa giác để bo chính xác theo viền ranh giới (boundary) thực tế của từng đối tượng một cách tỉ mỉ, tách biệt các thực thể xếp đè lên nhau. | Kiểm tra độ mịn và độ chính xác của đường biên đa giác; rà soát xem phần chồng lấp giữa các đối tượng có bị lẫn lộn pixel hay không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Quy tắc bảo vệ dữ liệu: chỉ sử dụng ba ảnh mẫu COCO công khai có sẵn và tuyệt đối không tải lên các hình ảnh cá nhân hay dữ liệu nội bộ riêng tư lên môi trường Colab công cộng.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi hay không rõ ràng, cần dừng lại ngay lập tức và báo cáo cho Lab Coach để được hỗ trợ giải quyết chứ không tự ý phán đoán gán nhãn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
