# 🛡️ NetGuard — NSL-KDD Data Engineering Dashboard
โครงงานกระบวนวิชา **204426 (เทอม 1/69)** — Data Engineering Pipeline สำหรับข้อมูล Network Intrusion (NSL-KDD)
แสดงผลด้วย **Streamlit** ในธีมแบบ dashboard (ส้ม-ขาว, sidebar เมนู, การ์ดมุมโค้ง)

## โครงสร้างโปรเจกต์
```
├── app.py            # Streamlit UI (8 หน้า)
├── pipeline.py       # โค้ด Data Engineering / โมเดล (ย้ายมาจาก notebook)
├── styles.py         # CSS ธีม
├── requirements.txt
├── .streamlit/config.toml
├── data/             # (ไม่บังคับ) KDDTrain+.txt, KDDTest+.txt
└── notebooks/attack_type_nslkdd_DE.ipynb
```

## หน้าในแอป ↔ หัวข้อรายงาน
| หน้า | เนื้อหา | บทในเล่มรายงาน |
|---|---|---|
| Dashboard | ภาพรวม dataset, class, data flow, ผลโมเดล | บทที่ 1, 4 |
| Pipeline | Diagram Input→Process→Output, step log, validation | บทที่ 2 |
| Data Quality | Profiling, Quality Report, correlation, skew, unseen attack | บทที่ 2–3 |
| Feature Eng. | ฟีเจอร์ที่สร้างเพิ่ม + เหตุผล | บทที่ 3 |
| Ablation | เทียบทีละขั้นตอน DE (เลือกด้วย CV บน train) | บทที่ 3 |
| Models | Dummy / LR / RF, report, confusion matrix, importance | บทที่ 3–4 |
| Error Analysis | R2L/U2R แยก seen/unseen subtype | บทที่ 4 |
| Predict | ทำนายทีละ connection จาก KDDTest+ | สาธิตโปรแกรม |

## รันในเครื่อง
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## ขึ้น GitHub + Deploy
```bash
git init
git add .
git commit -m "NSL-KDD DE dashboard"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
จากนั้นไปที่ https://share.streamlit.io → **New app** → เลือก repo, branch `main`, main file `app.py`

## หมายเหตุ
- ทุกการตัดสินใจ (dedup, constant, correlation, top services) คำนวณจาก **train เท่านั้น** เพื่อกัน data leakage
- Ablation/Models ใช้เวลารันหลายนาที (RF + CV) จึงมีปุ่ม **RUN** และ cache ผลไว้ — ลด `n_estimators` ใน Sidebar ถ้าต้องการให้เร็วขึ้น
- อยากให้เปิดเร็วบน Cloud: commit ไฟล์ข้อมูลไว้ใน `data/` (train ~19 MB, test ~3 MB)
