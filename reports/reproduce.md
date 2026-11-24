# Tái lập nghiên cứu

Dùng Python 3.12 và cài các phiên bản trong `requirements.txt`. Cấu hình huấn luyện sử dụng PyTorch CUDA khi thiết bị hỗ trợ. Chọn cùng trình thông dịch cho notebook.

Tải [bộ dữ liệu JPX](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data). Đặt `stock_prices.csv` và `financials.csv` của cả `train_files/` và `supplemental_files/` đúng dưới `data/raw/`.

Chạy từ thư mục gốc.

```powershell
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests

python scripts/run_jpx_project.py prepare
python scripts/run_jpx_project.py develop --pilot
python scripts/run_jpx_project.py develop
python scripts/run_jpx_project.py select
python scripts/run_jpx_project.py train
python scripts/run_jpx_project.py rank
python scripts/run_jpx_project.py evaluate
python scripts/run_jpx_project.py backtest
python scripts/verify_jpx_release.py
```

- Kết quả lưu vào `artifacts/`. `--pilot` chạy trọn một lượt. Lượt đầy đủ dùng lại kết quả pilot và các lượt hoàn tất khi mã nguồn, dữ liệu, thiết bị, phiên bản thư viện và tệp đầu ra khớp. `--raw-dir` và `--output` đổi đường dẫn.
- `select` khóa ứng viên, tham số, danh sách mô hình và số phiên validation. `rank` dùng giá và công bố chưa ghép nhãn để ghi `predictions.csv`. `evaluate` mới ghép `Target`. `backtest` tạo ba bộ bảng về chi phí và độ phủ.
- `prepare` lưu giá đã nối trong `artifacts/prices.pkl` và công bố đã nối trong `artifacts/financial.pkl`. `configs/lightgbm_finance.json` chứa các cột và tham số LightGBM đã dùng. Manifest và sổ từng lượt ghi cấu hình đã chạy.
- Script xác minh dựng mới trong thư mục tạm từ dữ liệu gốc và cấu hình đã khóa, so hạng chính xác và so bảng số theo dung sai. Để chạy lại cả quá trình chọn mô hình, bắt đầu từ `prepare` và `develop`. Nạp model và bảng pickle do chính lượt chạy tạo.
- Chạy lại năm notebook chính sau khi có các bảng. Những notebook tải về dùng để đọc phương pháp và đối chiếu nguồn.

Các phép kiểm độc lập chạy từ thư mục gốc. `audit_jpx_data.py` dùng toàn bộ gói dữ liệu giải nén.

```powershell
python scripts/audit_jpx_data.py --output data/interim/jpx_audit.json
python scripts/check_label_precision.py
python scripts/check_validation_signals.py
python scripts/check_validation_signals.py --augment-ohlcv
```

Hai phép thử tín hiệu cần kết quả `develop`. Lệnh đầu ghi `validation_signal_checks.csv`. Lệnh có `--augment-ohlcv` ghi `validation_ohlcv_augmentation.csv` và `validation_ohlcv_augmentation_folds.csv`. Cả ba bảng nằm ngay trong thư mục kết quả v2.

[Notebook chính](../notebooks/jpx_overview.ipynb) · [Kết quả](jpx_release.md)
