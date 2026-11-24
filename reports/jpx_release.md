# EquityRank · Từ dữ liệu JPX đến kết quả xếp hạng

EquityRank nghiên cứu liệu giá, khối lượng và công bố tài chính đã biết cuối phiên có giúp xếp hạng lợi nhuận cổ phiếu trong ngắn hạn. LightGBM finance dẫn đầu về Sharpe trong nhóm mô hình dùng để chọn trên validation. Trên tập kiểm thử, Sharpe gần bằng quy tắc đảo chiều trong phiên và mô phỏng ghi nhận lỗ sau chi phí ở mức phí 5 bps. [Notebook tổng quan](../notebooks/jpx_overview.ipynb) · [Lệnh tái lập](reproduce.md)

## Bài toán và đầu ra

Tại cuối phiên `t`, mô hình nhận một dòng đặc trưng cho mỗi mã đủ điều kiện. `Target[t]` đo lợi nhuận từ đóng cửa `t+1` đến đóng cửa `t+2` sau điều chỉnh chia/gộp. Các mốc tính theo lịch giao dịch JPX. Đầu ra là `Date, SecuritiesCode, Score, Rank`, trong đó hạng 0 có điểm dự báo cao nhất trong phiên. Hai đầu bảng được dùng để đo tín hiệu mua và bán khống. [Định nghĩa chỉ tiêu JPX](https://www.kaggle.com/code/smeitoma/jpx-competition-metric-definition)

## Dữ liệu và quyết định tiền xử lý

| Nguồn JPX | Số dòng | Vai trò |
| --- | ---: | --- |
| `stock_prices.csv` chính và bổ sung | 2.602.412 | Giá, khối lượng, hệ số chia/gộp và nhãn của 2.000 mã từ 04/01/2017 đến 24/06/2022. |
| `financials.csv` chính và bổ sung | 102.750 | Báo cáo thực tế, cập nhật dự báo và đính chính. |

**Giá và lịch.** Ghép hai tệp giá, kiểm tra khóa `(Date, SecuritiesCode)` và giữ lịch phiên chung. Có 8.426 dòng thiếu cả bốn giá OHLC và khối lượng bằng 0. Chỉ xếp hạng dòng có OHLCV hữu hạn và dương tại `t`. Các đặc trưng cửa sổ dùng giá quá khứ và hệ số chia/gộp đã có hiệu lực. Đối chiếu bằng giá điều chỉnh làm tròn đến 0,1 yên khớp 2.319.348 nhãn huấn luyện và 264.574 nhãn bổ sung trong những dòng kiểm tra được, với dung sai `1e-8`. Huấn luyện dùng `Target` do JPX cung cấp. [EDA giá](../notebooks/experiments/eda_stock_prices.ipynb) · [Mã đặc trưng](../src/features.py) · [Kiểm tra nhãn](../scripts/check_label_precision.py)

**Công bố đúng thời điểm.** Chuẩn hóa mã, kiểm tra thời điểm, trùng lặp và kỳ báo cáo. Tài liệu có hiệu lực từ phiên đầu sau ngày công bố. Mỗi tỷ số được cập nhật từ số thực tế hợp lệ và theo dõi riêng tuổi công bố, cờ thiếu. Bản đổi dự báo phát cờ sự kiện, còn bản đính chính chỉ cập nhật tỷ số khi ghép được với kỳ và phạm vi đã biết. Tập giá phụ, quyền chọn, giao dịch và các trường ngành/vốn hóa tĩnh nằm ngoài bộ đặc trưng. [EDA công bố](../notebooks/experiments/financial_disclosures.ipynb) · [Mã tiền xử lý](../src/release_data.py)

## Cấu hình và vai trò của các mô hình

Quy trình dùng sáu cửa sổ walk-forward. Nhãn vào train phải hoàn tất trước đầu mỗi cửa sổ. Validation của lượt chỉ nhận tín hiệu có `t+2` trước ranh giới tiếp theo. Tổng cộng có 700 phiên chấm chung. Bảy cấu hình có lưới nhỏ được chọn tham số trên hai đoạn bên trong train. Các ứng viên gồm Ridge, OLS, LightGBM, XGBoost, LambdaRank và mạng nơ-ron. Các đối chứng đảo chiều, động lượng và ngẫu nhiên giúp đặt mốc cho mô hình học máy.

**LightGBM finance** dùng 5 cột OHLCV và 9 đặc trưng công bố. Cấu hình đặt trước có 100 cây, tối đa 15 lá và tốc độ học 0,05. Nó đạt Sharpe cao nhất trong 26 ứng viên đủ điều kiện chọn. LightGBM finance và Ridge six được huấn luyện lại với nhãn hoàn tất chậm nhất 08/12/2021. Tập kiểm thử bắt đầu 09/12/2021 và đã từng được xem ở vòng đầu. Các biến thể phụ và lượt theo từng seed phục vụ phân tích trên validation. [Cấu hình LightGBM](../configs/lightgbm_finance.json) · [Mã chia tập](../src/release_experiment.py) · [So sánh mô hình](../notebooks/experiments/model_comparison.ipynb)

## Đánh giá thứ hạng và mô phỏng giao dịch

Spread mỗi phiên lấy lợi nhuận có trọng số của 200 mã đầu trừ lợi nhuận có trọng số của 200 mã cuối. Mỗi vế có trọng số giảm từ 2 đến 1 rồi chuẩn hóa về tổng 1. Ghép `Target` sau khi hạng đã chốt. Thiếu nhãn ở một mã thuộc hai vế thì spread phiên đó để trống. Sharpe kiểu JPX là spread trung bình chia cho độ lệch chuẩn mẫu, trước phí và chưa quy năm. IC kiểm tra tương quan thứ hạng trên toàn rổ. [Mã chấm](../src/evaluation.py)

| Phạm vi | Mô hình | Phiên chung | Sharpe | Spread (bps/phiên) |
| --- | --- | ---: | ---: | ---: |
| Validation | LightGBM finance | 700 | **0,1252** | +6,05 |
| Validation | Ridge six | 700 | 0,0768 | +10,39 |
| Kiểm thử | LightGBM finance | 123 | **0,0194** | +1,05 |
| Kiểm thử | Ridge six | 123 | −0,0726 | −16,94 |
| Kiểm thử | Đảo chiều trong phiên | 123 | 0,0178 | +2,22 |

Mô phỏng giao dịch dùng 130 phiên tín hiệu có `t+2`. Nó theo dõi giá vào, giá thoát, vị thế trễ và giá trị giao dịch để tính phí. Hai phép đối chiếu riêng lần lượt yêu cầu đủ giá cả rổ và dùng `Target` với tỷ trọng dự kiến. Ở 5 bps một chiều, ba cách cho **+47,34 bps trên 1 phiên chung đủ giá**, **−6,14 bps/phiên trên 125 phiên theo tỷ trọng dự kiến** và **−5,50 bps/phiên trên 131 phiên thực thi**. Các kết quả có mẫu và thời điểm ghi nhận khác nhau. Trong mô phỏng từng lệnh, LightGBM mở được 51.935/52.000 vị thế, 48 vị thế đóng trễ và 1 còn mở cuối kỳ. [Notebook thanh khoản và chi phí](../notebooks/experiments/liquidity_costs.ipynb)

## Lịch sử thực nghiệm và quyết định

| Giai đoạn | Kết quả và thay đổi |
| --- | --- |
| Vòng đầu | LightGBM finance với phép ghép cũ đạt Sharpe 0,1267 trên 700 phiên. Quy tắc chọn lúc đó dựa thêm vào chi phí trên 38 phiên đủ giá chung với Ridge, nên chọn Ridge six. Ridge đạt −0,0601 Sharpe và −14,05 bps/phiên trên 125 phiên kiểm thử. |
| Kiểm tra quy trình vòng đầu | Phép chạy lại dữ liệu mẫu được kiểm tra mốc nhãn, sau đó tái hiện 4.000 hạng trên hai phiên. Vòng này làm rõ rằng kết quả tái lập còn cần kiểm tra thời điểm dữ liệu được biết. |
| Vòng v2 | Tách tiêu chí chọn mô hình khỏi mẫu định giá hẹp. Sửa phép ghép công bố, thêm đối chứng kinh tế, học thứ hạng trực tiếp và kiểm tra neural theo seed. Chọn LightGBM finance theo Sharpe validation, rồi đo giả định giao dịch riêng. |
| Kiểm tra độ nhạy | Giữ lịch khi xảy ra sự cố 01/10/2020. Loại hai nhãn chịu ảnh hưởng khi huấn luyện Ridge làm Sharpe từ 0,0768 xuống 0,0760 trên cùng 700 phiên. |

Kết quả kiểm thử vòng đầu và v2 dùng tập ngày chung khác nhau. Lịch sử trên ghi lại vì sao tiêu chí chọn, đặc trưng và cách đánh giá được sửa qua từng vòng.

### Giá trị của các nhóm công bố

Trên 700 phiên chung, Sharpe của LightGBM OHLCV là **0,0775**. Thêm riêng tỷ số đạt **0,1058**, thêm tuổi và cờ thiếu đạt **0,0987**, thêm cờ sự kiện đạt **0,0904**, còn thêm cả nhóm đạt **0,1252**. Phép ghép cũ đạt **0,1151** trên cùng mẫu. Biến thể điền trung vị đạt **0,1305** và được đặt trong nhóm chẩn đoán từ đầu. [Đối chứng theo nhóm](../notebooks/experiments/financial_disclosures.ipynb)

### Thử thêm tín hiệu trên validation

Thay OHLCV bằng đặc trưng tương đối làm giảm cả Sharpe và spread, kể cả khi đi kèm công bố. Trộn điểm phân vị của LightGBM finance và đảo chiều trong phiên theo tỷ lệ 50/50 nâng spread lên **+7,42 bps/phiên** với Sharpe **0,0925**. [Mã thử](../scripts/check_validation_signals.py)

### Lượt cuối giữ OHLCV và thêm đặc trưng tương đối

| Nhóm thêm vào LightGBM finance | Sharpe | Spread (bps/phiên) | Lượt tăng cả hai chỉ số |
| --- | ---: | ---: | ---: |
| Trong phiên và qua đêm | 0,1157 | +6,31 | 3/6 |
| Lợi nhuận một phiên, biến động và thanh khoản | 0,1068 | +9,23 | 3/6 |
| Cả hai nhóm | 0,0900 | +7,14 | 2/6 |

Lượt cuối dùng cùng 700 phiên và cùng tham số LightGBM. Nhóm hoạt động giao dịch có khoảng bootstrap 95% cho chênh lệch spread từ **−2,13 đến +7,48 bps/phiên**, và cho chênh Sharpe từ **−0,0924 đến +0,0436**. Chốt cấu hình LightGBM finance đã chọn trên validation. [Lệnh tái lập các lượt thử](reproduce.md)

## Phát hiện rút ra

1. **Tuổi của công bố cũng mang thông tin.** Ghép riêng từng tỷ số nâng độ phủ đủ hai tỷ số từ 86,00% lên 95,04% trên 2.330.905 dòng giá hợp lệ trước mốc chia. Nhóm tuổi và cờ thiếu làm Sharpe tăng ngay cả khi chưa đưa tỷ số vào. Tuổi báo cáo biên lợi nhuận cũng đứng đầu trong phép hoán vị từng cột của LightGBM. Kết quả này gợi ý giữ cả giá trị lẫn độ mới của thông tin khi biểu diễn công bố. [EDA công bố](../notebooks/experiments/financial_disclosures.ipynb) · [Hoán vị đặc trưng](../notebooks/experiments/model_comparison.ipynb)
2. **Thêm spread có thể làm tăng dao động nhanh hơn.** Trên validation, LightGBM dẫn Ridge về Sharpe dù spread thấp hơn. Trong lượt cuối, thêm nhóm hoạt động giao dịch tăng spread từ +6,05 lên +9,23 bps/phiên, đồng thời nâng độ lệch chuẩn từ 48,32 lên 86,35 bps. Sharpe vì thế giảm. Hai chỉ số cần được báo cùng nhau.
3. **Đối chứng đơn giản thu hẹp kết luận trên kiểm thử.** LightGBM có Sharpe 0,0194 so với 0,0178 của đảo chiều trong phiên, với spread thấp hơn. Khoảng bootstrap ghép cặp 95% cho chênh Sharpe là −0,1964 đến +0,1880 theo khối 5 phiên. Hai ước lượng điểm cách nhau rất ít, còn độ bất định của mức chênh vẫn lớn.
4. **Khả năng khớp lệnh quyết định cách đọc kết quả.** Chỉ một phiên đủ giá toàn rổ cho cả hai mô hình. Mô phỏng từng lệnh cho LightGBM ghi nhận +2,72 bps/phiên ở mức phí 0 và −5,50 bps/phiên ở 5 bps một chiều, cùng giả định phí vay bằng 0. Lượng giao dịch và vị thế phải giữ lại có ảnh hưởng thực tế tới kết quả mô phỏng. Độ sâu thị trường, trượt giá, khả năng vay từng mã và cổ tức vẫn nằm ngoài mô hình thực thi.

## Xác minh

Hai mô hình cuối được dựng lại từ bốn tệp JPX gốc. Bảng LightGBM có **263.081 thứ hạng**, khớp từng dòng, và các bảng chấm điểm và chi phí khớp trong dung sai. Sổ sáu lượt phát triển giữ thông tin của lần sinh kết quả gốc. [Hướng dẫn tái lập](reproduce.md)
