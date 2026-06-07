import pandas as pd
import os
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor

BANKS = {
    "banka": {"uri": "bolt://localhost:4000", "path": "/Users/nikhilvasu/Downloads/Again_New/Bank_A"},
    "bankb": {"uri": "bolt://localhost:4001", "path": "/Users/nikhilvasu/Downloads/Again_New/Bank_B"},
    "bankc": {"uri": "bolt://localhost:4002", "path": "/Users/nikhilvasu/Downloads/Again_New/Bank_C"}
}

CHUNK_SIZE = 10000
print(f"Chunk size: {CHUNK_SIZE}")

def ingest_bank(bank_name, config):
    print(f"\n🚀 Starting {bank_name}...")
    driver = GraphDatabase.driver(config['uri'], auth=("", ""))
    path = config['path']

    # --- Accounts ---
    print(f"   👤 Loading Accounts for {bank_name}...")
    for chunk in pd.read_csv(os.path.join(path, "kyc.csv"), chunksize=CHUNK_SIZE):
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS row 
                MERGE (a:Account {id: row.entity_id}) 
                SET a.jurisdiction = row.jurisdiction,
                    a.industry = row.industry,
                    a.incorp_date = row.incorp_date,
                    a.is_shell = toFloat(row.is_shell),
                    a.dorm_days = toFloat(row.dorm_days),
                    a.device_entropy = toFloat(row.device_entropy)
            """, batch=chunk.to_dict('records'))

    # --- Documents ---
    print(f"   📄 Loading Documents for {bank_name}...")
    for chunk in pd.read_csv(os.path.join(path, "trade.csv"), chunksize=CHUNK_SIZE):
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS row
                MERGE (d:Document {id: row.doc_id})
                SET d.swift_ref = row.swift_msg_id,
                    d.commodity = row.commodity,
                    d.qty = toFloat(row.qty),
                    d.unit_price = toFloat(row.unit_price),
                    d.price_deviation = toFloat(row.price_deviation),
                    d.weight_gap_score =
                        CASE
                            WHEN toFloat(row.declared_weight_kg) = 0 THEN 0.0
                            ELSE toFloat(row.weight_gap_score) / toFloat(row.declared_weight_kg)
                        END
            """, batch=chunk.to_dict('records'))

    # --- SWIFT Transactions ---
    print(f"   💸 Loading SWIFT, Labels & Linking for {bank_name}...")
    for chunk in pd.read_csv(os.path.join(path, "swift.csv"), chunksize=CHUNK_SIZE):
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS row
                MERGE (s:Account {id: row.sender_id})
                MERGE (r:Account {id: row.receiver_id})
                MERGE (t:Transaction {id: row.msg_id})
                SET t.amount_usd = toFloat(row.amount_usd),
                    t.fx_deviation_pct = toFloat(row.fx_deviation_pct),
                    t.applied_exchange_rate = toFloat(row.applied_exchange_rate),
                    t.target_currency = row.target_currency,
                    t.pattern_type = row.pattern_type,
                    t.label = toInteger(row.label),
                    t.timestamp = row.timestamp
                MERGE (s)-[:SENDS]->(t)
                MERGE (t)-[:TO]->(r)
                WITH row, t
                MATCH (d:Document {swift_ref: row.msg_id})
                MERGE (t)-[:HAS_DOC]->(d)
            """, batch=chunk.to_dict('records'))

    driver.close()
    print(f"✅ {bank_name} finished successfully!")

if __name__ == "__main__":
    # Run all banks in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for b_name in ["banka", "bankb", "bankc"]:
            cfg = BANKS[b_name]
            if os.path.exists(cfg['path']):
                futures.append(executor.submit(ingest_bank, b_name, cfg))
            else:
                print(f"❌ ERROR: Cannot find folder '{cfg['path']}'!")

        # Wait for all to complete
        for future in futures:
            future.result()