import psycopg2
import time
import os

# Connect to your Supabase instance
conn = psycopg2.connect(
    host="aws-1-ap-southeast-2.pooler.supabase.com", 
    port="5432", 
    database="postgres",
    user="postgres.vnunjidwtvevzaqamezi",
    password="mAJORPROJECT@1234"
)
conn.autocommit = True

try:
    while True:
        # Clear the terminal screen for a dashboard feel (Mac/Linux)
        os.system('clear')
        print("🛡️  AML LIVE AUDIT MONITOR (Supabase Cloud) 🛡️")
        print("=" * 65)
        print(f"{'TRANSACTION ID':<20} | {'VERDICT':<10} | {'CONFIDENCE':<10} | {'TOPOLOGY'}")
        print("-" * 65)

        with conn.cursor() as cur:
            # Fetch the latest 15 alerts
            # Fetch the latest 15 alerts
            cur.execute("""
                SELECT transaction_id, verdict, confidence_score, str_metadata->>'pattern_detected' 
                FROM aml_audit_log 
                WHERE verdict IN ('FRAUD', 'WATCHLIST')
                ORDER BY logged_at DESC 
                LIMIT 15;
            """)
            rows = cur.fetchall()
            
            for row in rows:
                tx_id, verdict, conf, topology = row
                conf_str = f"{conf:.1f}%"
                
                if verdict == "FRAUD":
                    print(f"🚨 {tx_id:<17} | {verdict:<10} | {conf_str:<10} | {topology}")
                else:
                    print(f"⚠️  {tx_id:<17} | {verdict:<10} | {conf_str:<10} | {topology}")
                    
        print("\n(Press Ctrl+C to stop monitoring...)")
        time.sleep(3) # Refresh every 3 seconds

except KeyboardInterrupt:
    print("\n🛑 Closing monitor.")
finally:
    conn.close()