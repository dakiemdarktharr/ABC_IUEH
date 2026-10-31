# Rà soát mức sẵn sàng train Vast — 31/10/2026

**Kết luận: đủ để chuyển sang kiểm tra trên Linux bằng dữ liệu tổng hợp;
chưa đủ để bật toàn bộ chiến dịch train trả phí.** Lượt rà soát đã sửa các lỗi
ngân sách, seed, kiểm tra checkpoint và vòng đời tiến trình. Những việc phải
đo trên host thuê và đường đánh giá validation còn thiếu được ghi riêng bên dưới.
Không có bằng chứng để cam kết toàn bộ mục tiêu nghiên cứu sẽ xong trong $4.

Chỉ thị mới của người dùng là **$4 trần mềm all-in / $16 hạn mức cứng all-in**.
Điều này thay thế cấu hình cũ $16 mềm + $4 dự phòng = $20. Chạm dự báo $4 thì
dừng train và cần quyết định mới; phần chênh $12 không tự cho phép thêm fit,
retry hoặc kéo dài thời gian. Mọi chi phí setup, compute, resume, storage,
transfer, phí và xử lý cuối cùng thuộc cùng một chiến dịch.

## Các lỗi đã sửa

| Vấn đề tìm thấy | Thay đổi | Cách kiểm chứng |
|---|---|---|
| Bộ chạy và tài liệu dùng tổng $20, sai hạn mức mới | Đồng bộ planner, trainer, live gate, mẫu quote/budget và policy sang $4 mềm / $16 cứng | Test từ chối policy cũ, biên $4 và tổng vượt $16 |
| Mốc mềm chỉ xét core cost, bỏ qua một phần phí cuối | Xét tổng đã chi + compute + storage + egress + fees + xử lý cuối ngay ở mốc $4 | Test core < $4 nhưng tổng >= $4 vẫn bị từ chối |
| Bộ tính offline không làm tròn minimum/increment billing | Làm tròn lên, kiểm tra giá còn mới và storage bao phủ thời gian tồn tại instance | Test minimum billing, giá cũ/tương lai/thiếu và storage thiếu |
| Một fit có thể không hoàn tất trong reservation; chỉ dự báo fit đầu | Cần overhead đo thực; 5.000 update + mọi I/O + shutdown phải vừa reservation; kiểm tra toàn bộ 18 reservation và phần campaign còn lại | Test từ chối fit/campaign vượt thời gian hoặc dự báo tiền |
| Đổi seed train nhưng hai model hình học vẫn dùng seed khởi tạo cố định | Truyền seed của fit vào hidden initialization, giữ candidate/control cùng khởi tạo trong mỗi seed | Cùng seed khớp, khác seed khác; forward/backward ba arm hữu hạn |
| Giá mới có thể tính lại thời gian thuê đã qua ở mức thấp hơn | Khóa billing identity/rate trong lease và giữa các fit; tính tuổi snapshot vào elapsed time | Test thay rate, regression ledger, đổi boot và khôi phục clock |
| Bản backup local chưa đối chiếu đủ run ID và plan | Kiểm tra cả run, plan, fold, seed, model, code, data và số update | Test sidecar/payload hợp lệ nhưng thuộc run khác bị từ chối |
| Lệnh stop tùy chọn, SIGTERM có thể bỏ qua cleanup; I/O có thể chờ lâu | Bắt buộc instance ID và có stop client; cleanup trong finally; timeout gọi supervisor hợp tác; timer bao phủ cả campaign, gồm upload và xác minh | Test SIGTERM, lỗi, timeout, stop request và Linux timer/process group |
| Full suite có thể fork sau khi Torch đã được import | Kiểm thử supervisor thật dùng interpreter mới; unit test fork giả tách rõ khỏi kiểm thử thật | Test Linux riêng trong runtime CI |
| CI chính chưa chạy runtime Vast; các test thiếu package marker | Thêm package marker, lệnh kiểm tra code manifest/AST và job Linux/CPU | Chạy đúng cây code sẽ chuyển lên host; CI dùng cùng entry point |
| Test resume tổng hợp dùng pin CUDA dù CI cài PyTorch CPU | Đặt runtime identity riêng cho fixture CPU; giữ nguyên kiểm tra production CUDA ở bộ test runtime binding | Loop thực vẫn so sánh model/optimizer/sampler sau ngắt và resume |
| Pin và bản đóng gói v7 không còn khớp sau sửa | Tạo snapshot mới, manifest 74 file, runner/model pins mới; giữ snapshot cũ làm lịch sử | Đối chiếu SHA từng file và kiểm tra bản checkout mới |

## Kiểm tra đã chạy

- Suite nền: **442 passed**, 14 cảnh báo exporter ONNX cũ. Không thay code nền
  sau lượt kiểm tra này; cảnh báo không chứng minh trained export đã đạt.
- Suite staging rộng, gồm các draft validation và model geometry:
  **138 passed, 9 skipped, 27 subtests passed** trên Windows, CUDA bị tắt.
- Đúng gói runtime sẽ chuyển đi: **109 passed, 9 skipped, 27 subtests passed**;
  **74/74 file** khớp manifest, Python parse thành công. Các lượt bỏ qua do
  Linux process/timer, POSIX fsync hoặc quyền tạo liên kết của Windows được
  giữ rõ; không tính chúng là đã đạt trên Linux.
- Checkpoint/resume dùng dữ liệu tổng hợp, gồm model, optimizer, sampler và RNG.
  Chữ ký Ed25519, sai pin, backup lỗi, thứ tự fit và khả năng từ chối retry đều
  có kiểm tra. Đây là bằng chứng tính đúng của code, không phải accuracy.
- Job `vast-runtime-tests` mới chạy cùng gói trên Linux/CPU. Kết quả của đúng
  commit được kiểm tra trong GitHub Actions sau push; file JSON đi kèm ghi
  phạm vi local và các gate đích còn chưa xác minh.

Không thuê Vast, không train dữ liệu người, không chấm old dev5/final15 và
không gửi yêu cầu tới ban tổ chức trong lượt audit. Các file private giữ ngoài
Git. Dữ liệu fit-only không được xuất lại hoặc đổi run ID để làm mới lịch sử.

## Các điều kiện còn thiếu trước khi train trả phí

| Điều kiện | Vì sao chưa đóng được trong lượt local | Tiêu chí hoàn tất |
|---|---|---|
| Linux/CUDA trên chính host thuê | Windows không có WSL; Linux CPU CI không kiểm chứng driver, CUDA hay GPU thật | Cài pin đã ghi, `pip check`, freeze/hash; preflight RTX 3090 CC 8.6; synthetic CUDA update và checkpoint round-trip |
| Throughput, RAM/VRAM và overhead thực | Chưa có phép đo trên GPU định thuê; không điền số giả vào config | Đo bound bảo thủ gồm batch loading, init, save, copy-out, readback và verification; full schedule qua planner và ledger |
| Giá, tổng phí và ngừng tính tiền | Chưa có offer, account billing hoặc giới hạn provider được xác minh | Quote mới, ledger cộng dồn, đủ phí còn lại; stop client hoạt động; kiểm tra trạng thái provider và phương án dừng từ ngoài VM |
| Backup/recovery từ host thuê | Lịch sử mới có local synthetic Drive recovery; code archive mới trên Drive không thay cho kiểm tra training recovery | Trên host đích, upload/đọc lại/restore checkpoint và time journal tổng hợp; giữ latest head và trust pins ngoài VM; một writer |
| Validation dữ liệu người | Reader validation trong draft vẫn chủ động từ chối mở dữ liệu; chưa nối hoàn chỉnh khóa prediction trước target, comparator và analysis | Hoàn tất source binding và hai giai đoạn prediction/target, review rồi khóa protocol trước outcomes; chỉ dùng validation5 bên trong train15 |
| Độ tái lập Linux đầy đủ | Direct dependency pins chưa phải lock toàn bộ wheel Linux có hash | Lưu và rà soát môi trường resolved; chưa hứa byte replay qua host khác |
| Kết quả nghiên cứu và deployment | 18 fit là screen thăm dò; chưa có kết quả mới, trained ONNX/XR latency hay bằng chứng novelty | Đánh giá đúng protocol, uncertainty/tails và control; audit trained export sau khi có model; mục tiêu nghiên cứu mạnh hơn vẫn giữ nguyên |

Không thể bảo đảm hóa đơn dừng ở $16 chỉ bằng một callback local. Tài liệu
[billing của Vast](https://github.com/vast-ai/docs/blob/main/guides/reference/billing.mdx)
cho biết storage vẫn có thể tính phí khi instance dừng và thẻ đã lưu có thể
chi trả số dư âm. [Stop instance](https://github.com/vast-ai/docs/blob/main/guides/instances/manage-instances.mdx)
không đồng nghĩa ngừng mọi khoản phí. Vì vậy hard ceiling là ràng buộc phải
giữ khi quyết định thuê, không phải tính năng provider đã được chứng minh.

Giữ nguyên 400 -> 20, wxyz và mean geodesic endpoint20. Mâu thuẫn chính thức
20 mẫu / 100 Hz / 100 ms chưa được giải quyết. BOXRR vẫn đóng theo data-use
gate; Alyx-only chưa được xác nhận đủ điều kiện thi. Các hạn chế này không
được biến thành kết quả thực nghiệm hoặc khẳng định đã đăng ký/nộp bài.

## Bản để dùng tiếp

[Runtime mới](../research_checkpoints/vast_20261031/README.md),
[receipt dạng JSON](results/vast_runtime_review_20261031.json) và
[GitHub Actions](https://github.com/dakiemdarktharr/ABC_IUEH/actions).
Từ thư mục staging bên trong runtime, chạy
`python -m vast_training_v1.run_checks`. Cấu hình private đã giữ pin mới và
run IDs cũ; giá, đường dẫn đích và timing chưa biết vẫn để null để bị từ chối.

Rclone chỉ sao lưu gói code mới theo đường dẫn có SHA và kiểm tra bytes đọc
lại. [Copyto](https://rclone.org/commands/rclone_copyto/) không xóa file đích;
`--immutable` cùng readback giúp phát hiện sai bytes, không tạo khóa writer
hay bảo đảm độ bền vô thời hạn. Receipt đồng bộ nằm riêng ngoài Git.

Đã quan sát lỗi **Google Drive HTTP 403 quota** với OAuth client dùng chung của
rclone. Retry có giới hạn và đọc lại toàn bộ gói code sau đó đã đạt. Đây vẫn là
hạn chế vận hành: trước paid run cần kiểm tra quota trên host đích và cấu hình
client riêng hoặc phương án đã thử thay thế; không dùng ngân sách Vast để
giải quyết lỗi OAuth. Lưu bản code không thay thế kiểm tra khôi phục training.
