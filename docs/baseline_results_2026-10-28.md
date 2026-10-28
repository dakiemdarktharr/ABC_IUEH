# Baseline và features thực đo — 28/10/2026

Đây là pilot WhoIsAlyx trên **15 train / 5 development** users, không phải điểm
thi chính thức hoặc kết quả final holdout. Split15/5/15 và giả thuyết chính giữ
nguyên. [Cấu hình](../configs/baseline_features.json) đã ghi trước khi xem forecast
errors; [kế hoạch](baseline_feature_plan.md) mô tả mọi biến thể và cách fitting.

## Kết quả thực tế

Mean geodesic error tại output20, theo độ; thấp hơn tốt hơn. Cùng **3.330.312 train**
và **1.189.027 development** cửa sổ stride1 cho mọi phương pháp. CI95% dùng2.000
bootstrap theo user với ratio tổng lỗi/tổng cửa sổ, giữ toàn bộ windows của mỗi
user được lấy mẫu. Dev chỉ có5 user nên kết quả và CI là thăm dò; train là in-sample.
JSON kèm cả user-macro mean, không đánh đồng với window-weighted mean trong bảng.
Các CI có điều kiện trên các model/seed đã fit, không bao gồm toàn bộ bất định
do lấy lại tập train hoặc thay seed huấn luyện.

| Phương pháp | Mean train° | Mean dev° | P95 dev° | CI95% mean dev° |
|---|---:|---:|---:|---|
| ZV | 7.0575 | 7.2638 | 27.1383 | [6.2856, 8.2576] |
| CAV_W1 | 3.5697 | 4.1045 | 14.0410 | [3.5625, 4.6743] |
| CAV_W5 | 4.1050 | 4.6974 | 16.1460 | [4.0759, 5.3578] |
| CAV_W20 | 5.6020 | 6.2764 | 21.6739 | [5.4263, 7.2388] |
| equal_mixture | 3.9492 | 4.4099 | 15.4934 | [3.8238, 5.0885] |
| static_mixture | 3.3138 | 3.7249 | 13.1185 | [3.2423, 4.2704] |
| hard_h1_a0 | 3.5697 | 4.1045 | 14.0410 | [3.5625, 4.6743] |
| hard_h20_a0 | 3.6688 | 4.2774 | 14.6766 | [3.6946, 4.8864] |
| hard_h1_a19 | 3.5697 | 4.1045 | 14.0410 | [3.5625, 4.6743] |
| hard_h20_a20 | 3.6974 | 4.3138 | 14.8464 | [3.7159, 4.9436] |
| hard_complete_horizon | 3.5784 | 4.1275 | 14.1230 | [3.5781, 4.7048] |
| rank_h1_a0 | 3.7263 | 4.2587 | 14.6264 | [3.6916, 4.8672] |
| rank_h20_a0 | 3.7367 | 4.2701 | 14.7997 | [3.6932, 4.8968] |
| oracle_bank_hindsight | 3.0164 | 3.3409 | 11.7297 | [2.9092, 3.8162] |
| MLP_seed_20261026 | 3.4305 | 3.8964 | 11.7706 | [3.4114, 4.3823] |
| MLP_seed_20261027 | 3.5452 | 4.0574 | 12.0207 | [3.5621, 4.5536] |
| MLP_two_seed_mean_error | 3.4878 | 3.9769 | 11.6743 | [3.4868, 4.4680] |

ZV giữ nguyên hướng cuối. CAV_W1/W5/W20 dùng vận tốc góc từ1/5/20 khoảng mẫu quá
khứ. Static mixture được fit **chỉ trên65.536 train windows**, grid35 tổ hợp
simplex bước0,25, tối thiểu actual angle của quaternion trộn. CAV_W1 có lỗi dev
thấp nhất trong ba CAV đã định trước; đây chỉ là mô tả, không chọn hoặc promote
model/checkpoint cho final. Oracle nhìn đáp án tương lai để chọn expert tốt
nhất từng window: nó chỉ là diagnostic, không phải predictor nhân quả.

MLP là1197→64ReLU→60, **80.572 parameters**,2.000 Adam steps/seed,batch128,lr0,001.
Scaler3axis fit trên từng adjacent train increment một lần. Loss train là
sign-aligned quaternion MSE trên20 outputs; bảng đo endpoint20 angle thật.
Không early-stop hoặc chọn checkpoint theo dev; chưa chứng minh hội tụ hay
tối ưu hyperparameters. MLP_two_seed_mean_error là **trung bình lỗi** của hai
run, không phải prediction ensemble hay một model mới.

## Tín hiệu đối với hướng chính

| Contrast dev (h20 trừ đối chứng) | Mean delta° | CI95% paired user-cluster° | Users cải thiện |
|---|---:|---|---:|
| hard_h20_a0_minus_hard_h1_a0 | +0.1729 | [+0.1321, +0.2138] | 0/5 |
| hard_h20_a0_minus_hard_h1_a19 | +0.1729 | [+0.1321, +0.2138] | 0/5 |
| hard_h20_a0_minus_hard_complete_horizon | +0.1499 | [+0.1165, +0.1833] | 0/5 |

Hard routing bằng h20,a0 kém h1,a0 khoảng0,173°, không user dev nào cải thiện.
Đối chứng h1,a19 có cùng tuổi issue với h20,a0 vẫn cho kết luận bất lợi này.
Horizon h, target-age a và issue-age h+a không phải ba biến độc lập; các đối
chứng này không chứng minh đã tách ba causal effects riêng biệt.
Kết quả không ủng hộ giả định rằng chỉ đổi điểm chọn expert sang h20 sẽ tốt hơn.
Static train-fitted mixture là đối chứng mạnh hơn các hard routers trong lượt này.
Nó cũng có mean dev thấp hơn cả hai MLP seed trong ngân sách cố định này; chưa
được suy rộng thành kết luận về mọi neural architecture hoặc model đã hội tụ.
Oracle3,341° so với static mixture3,725° chỉ cho thấy headroom trong expert bank;
không chứng minh rằng predictor nhân quả có thể khai thác toàn bộ headroom đó.

Đây chưa phải thí nghiệm selector học được244parameters của hướng chính. Không
được dùng bảng này làm bằng chứng xác nhận hoặc bác bỏ hoàn toàn hypothesis của
selector đó. Bước tiếp theo là train các selector với cùng kiến trúc/data/compute
trên train, đánh giá dev cùng motion-only/static/Complete-Horizon; nếu h20 không
thắng thì thu hẹp claim hoặc giữ kết quả âm. Không sửa threshold, holdout hoặc
đổi mục tiêu sau khi xem final. Ngưỡng xác nhận đã định trước vẫn là>=3% và0,05°,
CI95% paired delta upper<0, ít nhất60% final users cải thiện.

## Features đã sẵn sàng

- Sáu biến thể10chiều: h1_a0,h20_a0,h1_a19,h20_a20,complete_horizon,motion_only.
- Mỗi evidence vector có4past expert errors,3speed norms,3velocity cosines;
  motion-only có10motion features như kế hoạch. R=80; toàn bộ evidence nằm trong
 400samples quá khứ; chỉ dùng errors có target đã xuất hiện.
- Origin399..N-21; target endpoint=t+20; không vượt session/gap. Future expert
  endpoint losses là labels riêng, không nằm trong feature vectors.
- Raw featuresfloat32 và train-only population scalers được lưu local; mỗi
  scaler fit3.330.312train rows, dùng nguyên cho dev. Analytic routing dùng raw
  scores; các scaler chuẩn bị cho selector học được, chưa dùng để fit selector.
- Cache **1,724,653,596 bytes**, dựng trong **339.42 giây**.

## Tài nguyên và kiểm tra

Chạy local bằng NumPy/CPU RyzenAI5340,RAM15,29GiB, BLAS/OMP/MKL giới hạn1thread;
không dùng RTX5050 hoặc dịch vụ trả phí. Analytic fitting/evaluation mất
**123.84 giây**; neural fits: **seed20261026: 27.27 giây, seed20261027: 24.60 giây**;
toàn neural stage gồm fitting/evaluation/IO là **789.95 giây**.
Đây là wall time trên desktop đang làm việc, không phải benchmark riêng biệt
hoặc latency targetXR. Hai dense layers có80.448matrix MACs/window; con số này
không gồm quaternion preprocessing/output/geometry và không phải tổng MAC toàn graph.

[Review độc lập](baseline_stage_review.md) kiểm code, replaySciPy và synthetic
tests; reviewer không đọc kết quả thật và không độc lập xác nhận efficacy.
Coordinator đối chiếu các receipt, số hàng, hashes và aggregate outcomes.
Toàn suite local đạt**159 synthetic tests**; MLP gradients, Adam và checkpoint
continuation có tests riêng. Chưa export
ONNX hoặc đo graph latency trên thiết bị XR trong stage này.

Source HDF5 và split SHA-256 trước/sau trùng nhau; final quaternion datasets
read=0 trong stage này. Không tạo final features/targets/errors hoặc fit trên
final users. SHA-256 đọc bytes toàn file chỉ nhằm kiểm integrity.

Private artifacts nằm ở `data/baseline_runs/20261028_r80/`: features.h5,
scalers.json, sealed feature_receipt.json, analytic_errors.h5,
analytic_user_statistics.json, static_mixture_fit.json và neural/ chứa weights,
scaler, errors và private user stats. Không file nào trong nhóm này được đưa lên
GitHub. Chỉ code/config/docs và [aggregate JSON](baseline_results_2026-10-28.json)
được công bố.

## Điều còn chưa xác nhận

[Source audit](data_sources.md) vẫn áp dụng: BOXRR không được dùng khi thiếu
recipient approval; Alyx-only challenge eligibility chưa được BTC xác nhận.
Mâu thuẫn20samples/100Hz/100ms và minimum actual context vẫn unresolved.
Không có foundation-model, SOTA, target-XR latency hoặc kết quả final holdout
được chứng minh ở đây. JEPA/EBM chưa chạy; ưu tiên hoàn thành kiểm chứng chính.
