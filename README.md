# BÁO CÁO ĐỒ ÁN: SO SÁNH CÁC MÔ HÌNH DỰ BÁO DÒNG CHẢY/LŨ LỤT
## Ứng dụng phương pháp Thống kê và Học sâu (Deep Learning)

---

## PHẦN 1: TỔNG QUAN VÀ CƠ SỞ LÝ THUYẾT TỪ CÁC NGHIÊN CỨU SOTA (STATE-OF-THE-ART)

Trong đồ án này, bên cạnh các mô hình cơ bản (ARIMAX, Vanilla LSTM), chúng tôi triển khai hai mô hình Học sâu tiên tiến dựa trên các công trình nghiên cứu uy tín thuộc lĩnh vực Thủy văn và Trí tuệ Nhân tạo. 

### 1. Mô hình EA-LSTM (Entity-Aware LSTM) - Hướng tiếp cận Kiến trúc
*   **Trích dẫn:** Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. (2019). *Towards improved predictions in ungauged basins: Exploiting the power of machine learning for large-sample hydrology.* Hydrology and Earth System Sciences, 23(12), 5089-5110.
*   **Mô tả:** Bài báo này đề xuất biến thể EA-LSTM nhằm giải quyết bài toán dự báo trên nhiều lưu vực khác nhau. Thay vì để mạng LSTM tự học mọi thứ từ chuỗi thời gian, nhóm tác giả Kratzert et al. đã thiết kế lại **cổng vào (Input Gate)** của LSTM. Cổng này được điều khiển trực tiếp bởi một vector đặc trưng tĩnh (Static Basin Attributes - ví dụ: diện tích lưu vực, độ dốc, loại đất). Điều này giúp mô hình "nhận thức" được thực thể (Entity-Aware), từ đó kiểm soát dòng thông tin khí tượng động (mưa, nhiệt độ) đi vào bộ nhớ tế bào ($C_t$) một cách phù hợp với đặc tính địa lý của từng trạm đo.

### 2. Mô hình Peak-Penalized LSTM - Hướng tiếp cận Hàm mục tiêu
*   **Trích dẫn:** Feng, D., Fang, K., & Shen, C. (2020). *Enhancing streamflow forecast and extracting insights using long short-term memory networks with data integration at continental scales.* Water Resources Research, 56(9).
*   **Mô tả:** Nghiên cứu này chỉ ra điểm yếu chí mạng của LSTM khi sử dụng hàm mất mát MSE (Mean Squared Error) truyền thống: mô hình thường có xu hướng dự báo trung bình hóa, dẫn đến việc "bỏ lỡ" (underestimate) các đỉnh lũ cực đoan. Để khắc phục, một **Hàm mất mát bất đối xứng (Asymmetric Peak-Penalized Loss)** được áp dụng. Hàm này nhân thêm một trọng số $\gamma$ vào các mẫu dữ liệu có lưu lượng thực tế cao, đồng thời trừng phạt nặng hơn khi mô hình dự báo thấp hơn thực tế. Điều này ép mạng nơ-ron phải nhạy cảm hơn với các sự kiện mưa bão gây lũ.

---

## PHẦN 2: PHÂN TÍCH VÀ ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH

Dựa trên kết quả chạy thực nghiệm, dưới đây là đánh giá chi tiết Điểm mạnh - Điểm yếu của từng mô hình:

### Bảng Tổng hợp Chỉ số (Validation)

| Mô hình | RMSE | MAE | NSE | KGE |
| :--- | :--- | :--- | :--- | :--- |
| 1. ARIMAX (Stat Base) | 0.0277 | 0.0110 | 0.9360 | 0.9680 |
| 2. Vanilla LSTM (DL Base)| 0.1582 | 0.1123 | -1.0816| 0.0861 |
| 3. EA-LSTM (Arch SOTA) | 0.1296 | 0.0956 | -0.3957| 0.3203 |
| 4. Peak-LSTM (Loss SOTA)| 0.1075 | 0.0729 | 0.0396 | 0.5675 |

> **[THAY THẾ BẰNG ẢNH BIỂU ĐỒ CỘT SO SÁNH NSE VÀ KGE CỦA BẠN]**
> ![Biểu đồ so sánh chỉ số NSE và KGE](<img width="622" height="303" alt="image" src="https://github.com/user-attachments/assets/160a5b80-7d4e-4012-aa22-a6a147ffd8e9" />
)

### Đánh giá Chi tiết (Tham chiếu Biểu đồ Hydrograph)

> **[THAY THẾ BẰNG ẢNH BIỂU ĐỒ HYDROGRAPH (ĐƯỜNG CONG THỦY VĂN) CỦA BẠN]**
> ![Đường cong Thủy văn Hydrograph]()

#### 1. ARIMAX (Đường nét chấm màu xám)
*   **Điểm mạnh:** Đạt chỉ số hoàn hảo ($NSE = 0.9360$). Trên biểu đồ, đường ARIMAX bám sát đường quan trắc thực tế (màu đen) gần như tuyệt đối ở cả mùa kiệt và đỉnh lũ.
*   **Điểm yếu:** Đây là hiện tượng đặc trưng của dự báo trượt 1 bước (one-step-ahead forecasting) trong thống kê học. Mô hình ARIMAX tận dụng trực tiếp giá trị thực tế của ngày hôm trước ($y_{t-1}$) để dự báo cho ngày hôm nay ($y_t$). Mặc dù sai số rất thấp, nhưng mô hình chủ yếu "sao chép" (shadowing) quá khứ (độ trễ pha 1 ngày) thay vì thực sự học được mối quan hệ nhân quả phức tạp phi tuyến từ lượng mưa sang dòng chảy. Nó sẽ thất bại nếu yêu cầu dự báo xa (multi-step ahead).

#### 2. Vanilla LSTM (Đường gạch chấm màu lục lam)
*   **Điểm mạnh:** Cấu trúc đơn giản, dễ triển khai bằng PyTorch.
*   **Điểm yếu:** Thất bại hoàn toàn ($NSE = -1.0816$). Nhìn vào biểu đồ, mô hình bị lệch nền (under-bias) nghiêm trọng, dự báo mùa kiệt rớt xuống mức 0.5 (trong khi thực tế là 0.75). Vào đợt lũ tháng 1-2/2004, đường dự báo phản ứng rất yếu và hoàn toàn không bắt được đỉnh lũ.

#### 3. EA-LSTM - Kratzert et al. (Đường nét đứt màu đỏ)
*   **Điểm mạnh:** Hình thái sóng (hydrograph dynamics) ổn định hơn Vanilla LSTM nhiều, KGE đạt 0.32. Hầu như không có hiện tượng nhiễu răng cưa (noise spikes).
*   **Điểm yếu:** Vẫn bị hội tụ vào "vùng an toàn", đường dự báo mùa cạn nằm dưới đường thực tế (~0.68 so với 0.75). Nhịp tăng lũ có xuất hiện nhưng bị "hụt hơi" (chỉ chạm mốc 0.95 thay vì 1.05). Cần bổ sung thêm cơ chế bù trừ độ lệch nền (bias correction).

#### 4. Peak-Penalized LSTM - Feng et al. (Đường nét liền màu xanh lá)
*   **Điểm mạnh:** Là mô hình Deep Learning tốt nhất trong thực nghiệm ($NSE = 0.0396, KGE = 0.5675$). Hàm loss chuyên biệt đã ép thành công mô hình bám sát đường cơ sở (~0.7) ở giai đoạn đầu và **đặc biệt là phản ứng cực mạnh với đỉnh lũ**. Nó là mô hình DL duy nhất theo kịp biên độ biến động lớn của lưu vực trong tháng 2/2004.
*   **Điểm yếu:** Phản ứng hơi thái quá (Over-shooting). Cuối tháng 12/2003 và tháng 1/2004, do tác dụng của hàm phạt (penalty), khi có tín hiệu mưa lớn, mô hình đã vọt lên quá sớm và vọt quá cao (chạm 1.18) trước khi lũ thực sự về.

---

## PHẦN 3: KẾT LUẬN

Qua đối sánh, ta thấy việc tích hợp kiến thức chuyên ngành (Domain Knowledge) vào mô hình Học sâu đóng vai trò sống còn. Việc tùy chỉnh Hàm mất mát nhắm vào đỉnh lũ (Peak-Penalized) tỏ ra hiệu quả nhất để khắc phục nhược điểm dự báo quá an toàn của LSTM truyền thống. Trong tương lai, sự kết hợp giữa **Kiến trúc EA-LSTM** và **Hàm mất mát Peak-Penalized** hứa hẹn sẽ mang lại một siêu mô hình toàn diện cho bài toán dự báo lũ lụt.
