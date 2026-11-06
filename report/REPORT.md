# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11/2026**

**Runtime Colab:** GPU

**Python: 3.13.16**
**PyTorch: 2.11.0+cu130**
**Ultralytics: 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

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

1. Record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh**. Do mô hình đang sử dụng là Classification - dự đoán nhãn cho toàn bộ ảnh chứ không phải cho từng vật thể. 
3. Danh sách tên lớp đi kèm với taxonomy/checkpoint
4. Nếu chỉ ghi mỗi tên lớp, ngữ cảnh sẽ bị thiếu và hệ thống sẽ không thể hiểu được chính xác hay phân biệt giữa các lĩnh vực. Ngoài ra việc việc truy vấn bằng id sẽ giúp hiệu năng tốt hơn.
5. Khi ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần nói rõ những điều cốt lõi sau: Xác định rõ kích thước và độ rõ nét, thứ tự ưu tiên giữa các lớp và tập trung vào ngữ cảnh tổng thể thay vì chú ý vào các chi tiết nhỏ. 
6. Vì `score` chỉ là ý kiến chủ quan của AI, còn ground truth bắt buộc phải là sự thật khách quan tuyệt đối và để có ground truth chuẩn, bắt buộc phải có sự can thiệp và kiểm định của con người.

## 2. Phát hiện vật thể

1. Record: `class_name`= person ; `score`= 0.610931 ; `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] ; `bbox_width`= 61.44 ; `bbox_height`= 47.59
2. Góc trên-trái: tọa độ ngang 0.12 pixel và tọa độ dọc 263.18 pixel, góc dưới-phải: tọa độ ngang 61.55 pixel và tọa độ dọc 310.78 pixel, chiều rộng là 61.44 pixel và chiều cao là 47.59 pixel.
3. Ngưỡng 0.20: 17 vật thể. Ngưỡng 0.60: 6 vật thể.
4. Khi hạ ngưỡng, người **reviewer** cần làm: Loại bỏ các hộp do mô hình nhận diện nhầm, điều chỉnh lại biên độ hộp và sửa lại tên nhãn.  
5. **Quy tắc box**: Người gán nhãn phải vẽ hộp bao vừa khít các cạnh ngoài cùng của vật thể. Không được để thừa khoảng trống (padding) quá 2 pixel ra ngoài rìa vật thể và không được cắt lẹm vào bên trong vật thể, trừ trường hợp vật thể đó bị che khuất hoặc bị cắt bởi rìa ảnh.
6.
- Vật thể phát hiện: Chiếc lò nướng bên phải (oven 0.63) bị cắt mất cạnh phải do nằm ngoài rìa ảnh, hoặc chiếc bát bowl 0.50 bị chiếc cốc cup 0.45 che khuất một phần phía trước.
- Phương án xử lý: Trường hợp này bắt buộc phải viết thành quy tắc cố định trong Guideline thay vì đưa lên hỏi Lab Coach (escalation).
- Vì sao? Bởi vì hiện tượng vật thể bị che khuất (occlusion) hoặc bị cắt rìa (truncation) xuất hiện ở hầu hết mọi bức ảnh trong thực tế bối cảnh tự nhiên. Nếu mỗi lần gặp vật bị che mà người gán nhãn lại đi hỏi Lab Coach, hệ thống quản lý dự án sẽ lập tức bị tắc nghẽn. Viết rõ quy tắc trong Guideline giúp người gán nhãn tự chủ làm việc nhanh chóng và nhất quán.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record: `instance_id`= kitchen-011 ; `class_name`= spoon ; `score`= 0.359147 ; số điểm polygon=`polygon_point_count`= 37 ; 3–5 điểm đầu của `polygon_xy`=  [529.0, 66.0], [529.0, 72.0], [528.0, 73.0]
2. So với Box vốn chỉ bao quanh phạm vi ngoài cùng của vật thể, Polygon cung cấp thêm hình dạng hình học thực tế và đường biên chính xác (pixel-level geometry) của vật thể đó.
3.
- Class-id là mã định danh duy nhất cho riêng thực thể cụ thể đó trong bức ảnh này. Nếu trong ảnh có 3 chiếc thìa, chúng sẽ chung class_id nhưng có 3 instance_id khác nhau.
- Tracking ID là mã dùng để định danh một vật thể xuyên suốt qua nhiều khung hình của một video. Còn instance_id ở đây chỉ có giá trị đóng gói độc lập trong phạm vi của một bức ảnh tĩnh
5. **Quy tắc vẽ biên mask**: Người gán nhãn phải chấm các điểm polygon dọc theo đúng đường ranh giới pixel phân tách giữa vật thể và nền. Điểm chấm không được lệch ra ngoài nền quá 1 pixel và không được cắt lẹm vào bên trong cấu trúc của vật thể.
6. 
- Vị trí phát hiện: Khu vực mặt bàn màu xanh (dining table 0.58) chứa rất nhiều chiếc bát (bowl 0.70, bowl 0.55, bowl 0.74) nằm xếp chồng, chạm cạnh và che khuất lẫn nhau liên tục.
- Phương án xử lý: Trường hợp này nên phối hợp cả Guideline lẫn Escalation (Hỏi Lab Coach) theo lộ trình:
	• Trước hết, tra cứu Guideline: Guideline cần có sẵn quy tắc nền tảng về việc xử lý vật chồng lấp (ví dụ: "Khi hai vật chạm nhau, vẽ đường biên cắt ngang qua điểm tiếp xúc; vật nằm trước vẽ đè lên vật nằm sau"). Người gán nhãn áp dụng quy tắc này để tự xử lý 80% trường hợp thông thường.
	• Sau đó, Escalation (Hỏi Lab Coach) cho các Edge Case đặc biệt: Nếu có những vùng biên quá mờ do đổ bóng, hoặc chiếc bát bị che khuất đến mức mắt người không thể đoán được hình dạng bên dưới của nó (Amodal segmentation), lúc này người gán nhãn cần đưa lên hỏi Lab Coach.
- Vì sao? Vì bài toán phân đoạn (Segmentation) nhạy cảm với từng pixel. Nếu tự đoán mò đường biên ở các vùng chồng lấp quá phức tạp, dữ liệu ground truth sẽ bị nhiễu nặng, khiến mô hình AI sau này bị phân đoạn lem nhem. Lab Coach sẽ đưa ra quyết định chuẩn cuối cùng cho các trường hợp mập mờ này để cập nhật ngược lại vào tài liệu Guideline.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn chữ duy nhất đại diện cho toàn bộ bối cảnh của bức ảnh. | Ảnh có quá nhiều chủ đề: Ảnh chứa đồng thời cả xe buýt, xe khách và taxi mà không có xe nào chiếm ưu thế, dẫn đến việc mô hình chỉ đoán được lớp cab với độ tự tin thấp (score: 0.51). | Xem toàn bộ bức ảnh, tra cứu tiêu chí bối cảnh chính trong guideline để chọn ra đúng 1 nhãn đại diện tốt nhất. | Xem nhãn chữ được gán xem có khớp với ngữ cảnh tổng thể không; kiểm tra xem annotator có chọn đúng lớp ưu tiên khi ảnh hỗn hợp không. |
| Phát hiện vật thể | Tọa độ hình hộp chữ nhật ôm sát vật thể kèm nhãn lớp. | Sai lệch tọa độ và sót vật: Trong file JSON bổ sung, tọa độ dọc ymin, ymax (263.18, 310.78) bị lệch xuống nửa dưới của ảnh thay vì khớp vị trí người ngồi bên trái. Ngoài ra, nhiều thớt, đĩa nhỏ trên bàn bị bỏ sót. | Dùng công cụ kéo khung bao chữ nhật ôm vừa khít từng vật thể độc lập và gán nhãn chính xác cho từng hộp. | Kiểm tra xem các cạnh của hộp có bị lẹm hay thừa quá quy định không; rà soát xem có hộp nào bị lệch tọa độ trục (giống lỗi JSON) hoặc gán sai nhãn không. |
| Instance segmentation | Danh sách các tọa độ điểm (Polygon) tạo thành một mặt nạ (mask) bọc khít biên vật thể. | Biên mờ và vật đè nhau: Nhiều chiếc bát (bowl 0.70, bowl 0.55) trên bàn màu xanh xếp chồng và đè lên nhau, đường biên pixel cực kỳ mờ, khó bám cấu trúc chính xác cho từng cá thể (instance_id). | Dùng công cụ chấm điểm liên tiếp (Polygon) để vẽ một đường viền khép kín, bọc khít xung quanh ranh giới pixel thực tế của từng vật. | Soi kỹ từng pixel viền xem mask có bị lem ra nền (background) không, kiểm tra các vùng vật chạm nhau xem đường cắt biên có mạch lạc và đúng thực tế không. |

## 5. An toàn dữ liệu

1. 1. Quy tắc bảo vệ dữ liệu cho bài lab này là tuyệt đối bảo mật, không sao chép, chia sẻ hoặc sử dụng các tệp tin hình ảnh và bản ghi dữ liệu JSON của dự án ra bên ngoài phạm vi phòng lab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi hoặc phát hiện nghi vấn về an toàn thông tin, bạn sẽ dừng lại ngay lập tức và báo cáo trực tiếp cho Lab Coach (hoặc Quản lý dự án / Project Manager) để nhận hướng dẫn xử lý.

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
