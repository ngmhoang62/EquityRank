# EquityRank · Xếp hạng cổ phiếu JPX

Nghiên cứu xem giá, khối lượng và công bố tài chính có giúp phân biệt các cổ phiếu sẽ tăng hoặc giảm tương đối trong ngắn hạn hay không. Mỗi phiên, mô hình gán điểm cho từng mã đủ điều kiện rồi xếp hạng theo điểm. [Dữ liệu JPX](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)

## Đầu vào, đầu ra và mục tiêu

| Thành phần | Ý nghĩa |
| --- | --- |
| Đầu vào tại phiên `t` | OHLCV, hệ số chia/gộp, lịch sử trước phiên và công bố tài chính đã có hiệu lực. |
| Nhãn `Target[t]` | Lợi nhuận giữa giá đóng cửa đã điều chỉnh ở `t+1` và `t+2`, theo lịch phiên gốc. |
| Đầu ra | `Date, SecuritiesCode, Score, Rank`. Mỗi phiên xếp các mã có giá hữu hạn, dương và khối lượng dương. Hạng 0 là điểm cao nhất. |
| Tiêu chí chính | Sharpe theo ngày của spread trước phí giữa 200 mã đầu và 200 mã cuối. Trọng số mỗi vế giảm từ 2 đến 1 rồi chuẩn hóa. |

## Khám phá và xử lý dữ liệu

- Ghép **2.602.412 dòng giá** của **2.000 mã** từ 04/01/2017 đến 24/06/2022. Có **8.426 dòng** thiếu cả bốn giá OHLC, đều có khối lượng bằng 0. Khảo sát độ dài lịch sử, nhãn cực trị và sự cố ngày 01/10/2020. Giữ lịch gốc và đo riêng độ nhạy của hai ngày tín hiệu bị ảnh hưởng. [EDA giá](notebooks/experiments/eda_stock_prices.ipynb)
- Tạo đặc trưng lợi nhuận điều chỉnh, biến động, biên độ, thanh khoản và phân vị theo phiên. Hai tỷ số tài chính có lịch sử và tuổi riêng. Một bản tin dự báo được ghi nhận như một sự kiện và giữ nguyên tỷ số thực tế đã biết. [EDA công bố tài chính](notebooks/experiments/financial_disclosures.ipynb)

## Huấn luyện và đánh giá

- Phát triển bằng sáu cửa sổ walk-forward. Mỗi lần huấn luyện dùng nhãn đã hoàn tất trước cửa sổ cần dự báo. Tinh chỉnh qua hai cửa sổ trong tập huấn luyện, chọn cấu hình theo Sharpe trên tập xác thực rồi chấm trên **tập kiểm thử từ 09/12/2021**. Kết quả kiểm thử đã từng được xem ở vòng trước.
- So sánh Ridge, quy tắc đảo chiều và động lượng, điểm ngẫu nhiên, OLS, LightGBM, XGBoost, LambdaRank hai vế và MLP/TCN/GRU. Đo riêng IC, lợi nhuận theo nhóm hạng, hai vế, độ ổn định theo thời gian và khoảng bất định bootstrap. [So sánh mô hình](notebooks/experiments/model_comparison.ipynb)
- Mô phỏng chi phí theo vị thế vào được, vị thế thoát trễ và phần tiền mặt. So với phép tính chỉ dùng phiên đủ giá và phép ước tính dựa trên `Target`. Ba cách dùng mẫu số và thời điểm ghi nhận khác nhau. [Thanh khoản và chi phí](notebooks/experiments/liquidity_costs.ipynb)

## Kết quả

| Phạm vi | Mô hình | Phiên chung | Sharpe | Spread (bps/phiên) |
| --- | --- | ---: | ---: | ---: |
| Phát triển | LightGBM finance | 700 | 0,1252 | +6,05 |
| Phát triển | Ridge six | 700 | 0,0768 | +10,39 |
| Kiểm thử | LightGBM finance | 123 | 0,0194 | +1,05 |
| Kiểm thử | Ridge six | 123 | −0,0726 | −16,94 |
| Kiểm thử | Đảo chiều trong phiên | 123 | 0,0178 | +2,22 |

- **Công bố tài chính.** Tỷ lệ có đủ hai tỷ số trước mốc khóa tăng từ **86,00%** lên **95,04%**. Hiệu quả dự báo được đo bằng đối chứng cùng mẫu ở [báo cáo kết quả](reports/jpx_release.md).
- **Độ chắc chắn.** Trên tập phát triển, chênh lệch Sharpe giữa LightGBM finance và Ridge six là **+0,0484**. Khoảng bootstrap 95% ghép cặp theo khối 5 phiên chạy từ **−0,0234 đến +0,1337**, nên thứ tự hai mô hình còn bất định. [Báo cáo kết quả](reports/jpx_release.md)
- **Khớp lệnh và chi phí.** Trên tập kiểm thử, mô phỏng của LightGBM finance vào được **51.935** trên **52.000** vị thế dự kiến. Ở mức phí 5 bps trên giá trị giao dịch, lợi nhuận ròng đã ghi nhận là **−5,50 bps mỗi phiên mô phỏng**, với một vị thế còn mở cuối kỳ. [Thanh khoản và chi phí](notebooks/experiments/liquidity_costs.ipynb)
- **Tái lập.** Quy trình dựng lại từ dữ liệu gốc cho cùng **263.081** hạng. [Hướng dẫn](reports/reproduce.md)

## Công cụ và cách chạy

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, PyTorch, Matplotlib và Jupyter. [Notebook chính](notebooks/jpx_overview.ipynb) · [Tái lập](reports/reproduce.md) · [Dữ liệu, quyết định và kết quả](reports/jpx_release.md) · [Notebook tham khảo](notebooks/references/README.md)
