import time
import httpx

def get_status_safe(retries=3, timeout=15.0):
    for attempt in range(retries):
        try:
            r = httpx.get("http://localhost:8000/datasets/status/touche", timeout=timeout)
            
            # التأكد من أن السيرفر رد بكود نجاح 200
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    print(f"⚠️ السيرفر رد بنجاح ولكن الاستجابة ليست JSON! النص المستلم: {r.text[:100]}")
                    return None
            else:
                print(f"⚠️ السيرفر رد بكود خطأ ({r.status_code}). النص: {r.text[:100]}")
                
        except (httpx.ReadTimeout, httpx.ConnectTimeout):
            print(f" ⚠️ محاولة {attempt+1}/{retries} تأخرت أو تعذرت — إعادة المحاولة...")
            time.sleep(3)
        except httpx.RequestError as e:
            print(f"❌ خطأ في الاتصال بالسيرفر: {e}")
            time.sleep(3)
            
    return None

print("🔄 بدء مراقبة الفهرسة التلقائية...")
while True:
    s = get_status_safe()
    if s is None:
        print("⚠️ لم نتمكن من جلب الحالة الآن — سأحاول مجدداً بعد 10 ثوانٍ.")
    else:
        # التأكد من وجود المفاتيح المتوقعة في الـ JSON
        status = s.get('status', 'unknown')
        progress = s.get('progress', 0.0)
        message = s.get('message', '')
        
        print(f"📊 {status} | التقدّم: {progress*100:.1f}% | {message}")
        
        if status in ("done", "error"):
            if "error" in s and s["error"]:
                print(f"❌ تفاصيل الخطأ من السيرفر: {s['error']}")
            break
            
    time.sleep(10)