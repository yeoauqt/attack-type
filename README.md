# NSL-KDD Data Engineering Dashboard (Streamlit)

เว็บสาธิต Data Engineering Pipeline จาก `attack_type_nslkdd_DE.ipynb` (204426 เทอม 1/69)

## วิธีรัน
```bash
pip install -r requirements.txt
streamlit run app.py
```
- เปิดเว็บได้ทันที เพราะมีผลที่คำนวณไว้แล้วใน `artifacts/`
- ถ้าลบ `artifacts/` เว็บจะรัน pipeline ใหม่เองครั้งแรก (~4-5 นาที) หรือรัน `python build_artifacts.py`
- ถ้าลบ `data/` โค้ดจะโหลด NSL-KDD จาก GitHub ให้อัตโนมัติ (ต้องต่ออินเทอร์เน็ต)

## โครงสร้าง
| ไฟล์ | หน้าที่ |
|---|---|
| `pipeline.py` | ฟังก์ชัน DE ทั้งหมดจากโน้ตบุ๊ก (extraction, validation, cleaning, feature selection/engineering, ablation, โมเดล) |
| `app.py` | หน้าเว็บ 5 หน้า |
| `build_artifacts.py` | คำนวณ ablation + เทรนโมเดล เก็บลง `artifacts/` |
| `sample_upload.txt` | ไฟล์ตัวอย่างสำหรับทดสอบหน้า "ทำนายข้อมูลใหม่" (300 แถว) |

## หน้าเว็บ
1. ภาพรวม Pipeline (diagram Input → Process → Output + ตารางเทคนิค/วัตถุประสงค์)
2. Data Profiling & Quality Report
3. Pipeline Explorer (step log, correlated pairs, heatmap, ก่อน–หลัง transform)
4. Ablation & Model (เลือก setup ด้วย CV บน train, เทียบโมเดล, confusion matrix, error analysis)
5. ทำนายข้อมูลใหม่ (validate → transform → predict → ดาวน์โหลดผล)

## Deploy ขึ้น Streamlit Community Cloud
1. push repo นี้ขึ้น GitHub (public)
2. เข้า https://share.streamlit.io → New app → เลือก repo, branch `main`, Main file path = `app.py`
3. กด Deploy (ครั้งแรกจะโหลดข้อมูล NSL-KDD จาก GitHub อัตโนมัติ)

> ถ้าเวอร์ชัน scikit-learn บนเซิร์ฟเวอร์ไม่ตรงกับตอน build (1.8.0) เว็บจะรัน pipeline ใหม่เองครั้งแรก (~4-5 นาที)
