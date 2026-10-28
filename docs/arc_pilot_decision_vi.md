# Quyết định sau pilot GPU — 28/10/2026

**Chốt: chưa đủ cơ sở viết paper đề xuất kiến trúc exposure-distribution như một đóng góp mới có ưu thế thực dụng.** Có kết quả dự đoán dương rõ trên pilot: mô hình học được chuyển động tốt hơn static mixture. Nhưng phần distribution chưa tạo lợi ích đủ lớn trước các đối chứng trực tiếp; vector residual là model tốt nhất đã chạy và được giữ làm baseline kỹ thuật mạnh.

Không đổi vector từ nhánh đối chứng thành primary sau khi xem kết quả. Không coi residual, body-frame hoặc TCN tự thân là novelty. Không bảo đảm được nhận vào kỷ yếu từ một pilot.

## Thực nghiệm thật đã hoàn thành

- RTX 5050 Laptop, PyTorch 2.9.1+cu128, CUDA thực; 30 fits = 3 folds × 2 seeds × 5 nhánh.
- Chỉ dùng 15 user thuộc train pool cũ. Mỗi fold fit trên 10 người, validation trên 5 người khác; mỗi người làm validation đúng một lần. Tổng 3.330.312 origins được chấm trong mỗi scope.
- Dev cũ 5 người và final 15 người không được chấm hoặc đưa vào fitting/pretraining. Dữ liệu và split giữ nguyên fingerprint.
- Primary: cùng 5.000 Adam updates, B128, lịch lấy mẫu và encoder khởi tạo được ghép cặp. Secondary: checkpoint prefix chọn chỉ bằng thời gian, không dùng độ chính xác, trên cùng validation origins.
- Hai seed được tổng hợp bằng trung bình **error** ghép cặp; không phải ensemble hai prediction.

| Model | Cùng 5.000 bước: mean endpoint20° | Ngân sách thời gian tương đương° | Parameters |
|---|---:|---:|---:|
| Static mixture fit riêng từng fold | 3,313769 | 3,313769 | 4 hệ số fit bằng grid |
| NLL exposure + median | 2,877588 | 2,876846 | 1.717 |
| Direct scalar-angle | 2,883012 | 2,973867 | 1.649 |
| Angular-risk21 | 2,891339 | 2,891339 | 1.717 |
| Scalar-pinball | 2,903430 | 2,904588 | 1.649 |
| Direct vector-angle | **2,528655** | **2,580291** | 1.683 |

Đây là inner-validation trên train pool, không phải kết quả trên 5 dev user của bài cũ. Không lấy số ở hai partition khác nhau để tính một phần trăm cải thiện.

## Điều được dữ liệu hỗ trợ

NLL-median giảm 0,436181° (13,16%) so static; paired user-bootstrap CI95% của NLL−static là [−0,481902; −0,388640]°. Cả 15 user cải thiện. Khả năng học dự đoán từ lịch sử chuyển động đã có tín hiệu dương trong phạm vi pilot này.

So direct scalar, lợi thế NLL chỉ 0,005425° (0,19%), CI [−0,012786; +0,002140]° và chiều khác nhau giữa hai seed. So angular-risk, lợi thế là 0,013751° (0,48%); so pinball là 0,025842° (0,89%). Các mức này không đạt ngưỡng thực dụng đã đặt trước là ít nhất 3% **và** 0,05° trước mọi đối chứng bắt buộc. Gate primary thất bại.

Ở ngân sách thời gian tương đương, NLL vượt scalar 3,26%, nhưng vẫn không đạt hiệu ứng cần thiết trước risk21/pinball. Gate secondary cũng thất bại. Không thể bỏ các đối chứng không thuận lợi để gọi pilot thành công.

NLL không vượt vector ở bất kỳ validation user nào; vector có pooled mean error thấp hơn trong cả hai cách ghép compute. Contrast vector−static được tính thêm **hậu nghiệm**, không thay primary: −0,785115° (23,69%), CI [−0,882151; −0,684214]°; khi ghép thời gian là −0,733478° (22,13%), CI [−0,828338; −0,632844]°. Hai seed và ba fold đều cùng chiều. Đây là một lead kỹ thuật tốt; chưa phải xác nhận cho một hypothesis mới được đặt trước.

## Những suy luận chưa được phép

- Không chứng minh mọi distribution/JEPA/EBM đều không phù hợp. Chỉ một exposure-label model, encoder và recipe hữu hạn được kiểm tra.
- Không chứng minh khác biệt do riêng quyền đổi trục. Vector có action space/parameterization khác; cần ablation cô lập trước khi gán nguyên nhân.
- Median của nhãn nearest-on-arc không luôn tối ưu góc với target ngoài arc. Phản ví dụ SO(3) đã được ghi trước fitting; không có định lý Bayes mới được chứng minh ở đây.
- Train curves là diagnostics hữu hạn, không chứng minh hội tụ toàn cục. Loss khác nhau có scale/độ khó tối ưu khác nhau.
- Các CI có điều kiện trên hai seed và các CV fits cố định; không bao phủ mọi training randomness, dataset hay population. Inner folds còn là selection folds trong một chương trình đã xem dữ liệu.

## Compute và khả thi

Tổng synchronized training của 30 fits: 1.238,49 giây, khoảng 20,64 phút. Tổng thời gian các CUDA-context sessions, gồm chuẩn bị/validation/I/O: 3.914,56 giây, khoảng 1,09 giờ. Không thuê GPU.

Peak Torch CUDA allocated/reserved của toàn worker, gồm cache và các nhánh cùng resident: 452,57/486,00 MiB; không phải VRAM của riêng network. Hai lần dừng theo guard RAM được resume từ checkpoint. Một thao tác thu gọn working set của chính tiến trình đã giải phóng trang resident; không đổi tensor, optimizer, model, lịch batch hay ngưỡng bảo vệ. Thời gian đo có ảnh hưởng của paging.

Cùng số bước không cho thời gian bằng nhau: ratio lớn nhất/nhỏ nhất là 2,343 toàn bộ, 2,361 trong một fold/seed. Phép ghép thời gian riêng có ratio 1,0267 toàn bộ và 1,0383 ở cell xấu nhất, với độ hạt 100 bước. Đây là ngân sách thời gian xấp xỉ trên backend đã đo, không phải bằng nhau tuyệt đối về FLOPs. Static grid là baseline rẻ riêng, không có cùng optimizer budget với neural fits.

## Quyết định và việc giữ lại

1. Dừng narrative “distribution mới tốt hơn direct prediction” cho paper đề xuất model hiện tại. Giữ toàn bộ kết quả dương/âm và primary gate đã thất bại.
2. Giữ `vector_angle` làm baseline chính cho vòng nghiên cứu tiếp. Kết quả đủ tốt để bảo lưu implementation và tiếp tục kiểm chứng kỹ thuật; chưa đủ novelty để đổi tên nó thành đóng góp kiến trúc mới.
3. Nếu theo đuổi một bài phân tích/benchmark, cần đóng các khoảng trống đã biết: DESP/quaternion filter mạnh, controls cô lập action-space, kiểm tra trained batch1/ONNX/clean-Linux và bằng chứng đánh giá đã khóa. Một paper mới có novelty cao cần một cơ chế/câu hỏi khoa học khác được nearest-work review bảo vệ.

Source/data-use vẫn là điều kiện riêng: WhoIsAlyx chỉ ở phạm vi pilot phi thương mại có điều kiện; BOXRR còn khóa theo recipient-review gate; Alyx-only challenge eligibility chưa được BTC xác nhận. Mâu thuẫn 20 samples / 100 Hz / 100 ms vẫn unresolved. Không có final/hidden-test result hoặc target-XR latency.

## Hồ sơ kiểm chứng

[Plan](arc_pilot_plan.md), [toàn bộ kết quả](arc_pilot_results_2026-10-28.md), [aggregate](results/arc_pilot_20261028.json), [independent audit](arc_pilot_results_audit.md), [audit aggregate](results/arc_pilot_independent_audit.json).

Code được đo ở commit `13c74071156b5b8dfc893f42107ad2cb82ea6921`; raw worktree bytes đã lưu riêng, có mapping LF/CRLF. Sau khi pilot và audit hoàn thành, hai guard xác minh manifest/reference khi resume được sửa riêng; sửa này không thay model/loss/schedule và không chạy lại efficacy. Xem [closure](arc_pilot_code_review.md).

Nearest mechanisms đã được khảo sát gồm [Illahi/MMSys2023](https://doi.org/10.1145/3587819.3590972), [SG-nCDE/ICCV2025](https://bastianlb.github.io/forecasting-rotational-dynamics/), [Zhong/VRST2025](https://doi.org/10.1145/3756884.3765973), và [HAP2026](https://arxiv.org/html/2609.18548v1). Chúng giới hạn claim novelty của residual, rotation geometry, filtering và CV gating; survey có giới hạn, không chứng minh không còn prior khác.
