"""รันครั้งเดียวเพื่อคำนวณ ablation/โมเดล แล้วเก็บไว้ใน artifacts/ (ทำให้เว็บเปิดเร็ว)"""
import joblib, os, time
from pipeline import run_all
t = time.time()
art = run_all(lambda m, p: print(f"[{p:>4.0%}] {m}"))
os.makedirs("artifacts", exist_ok=True)
joblib.dump(art, "artifacts/artifacts.joblib", compress=3)
print(f"บันทึกแล้ว ({time.time()-t:.0f}s) best setup =", art["cfg"]["best_setup"])
