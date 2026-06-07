import json
import torch
import numpy as np
import psycopg2
from kafka import KafkaConsumer
from neo4j import GraphDatabase
from model import TBML_DetectionModel 
import hashlib
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
MEMGRAPH_URIS = {
    'A': "bolt://localhost:4000",
    'B': "bolt://localhost:4001",
    'C': "bolt://localhost:4002"
}
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""
KAFKA_TOPIC = 'incoming_swift_messages'

PG_HOST = "localhost"
PG_PORT = "5432"
PG_DB = "postgres"
PG_USER = "postgres"
PG_PASS = "password"

# ==========================================
# PHASE 1: SYSTEM PRE-FLIGHT
# ==========================================
print("⚙️ Booting Real-Time Deep AML Inference Engine...")

model = TBML_DetectionModel(hidden_dim=64, lstm_hidden=64, classes=3)

try:
    print("📂 Loading weights from global_model_final.npz...")
    npz_data = np.load('global_model_final.npz') 
    
    state_dict = model.state_dict()
    loaded_weights = [npz_data[f"arr_{i}"] for i in range(len(state_dict))]
    
    for i, key in enumerate(state_dict.keys()):
        state_dict[key] = torch.tensor(loaded_weights[i])
        
    model.load_state_dict(state_dict)
    model.eval() 
    print("✅ Multi-Modal Neural Network Weights Successfully Injected.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load weights. Error: {e}")
    exit()

print("⚙️ Connecting to Federated Banking Network...")
db_drivers = {
    bank: GraphDatabase.driver(uri, auth=(MEMGRAPH_USER, MEMGRAPH_PASS)) 
    for bank, uri in MEMGRAPH_URIS.items()
}
print("✅ Connected to Memgraph Nodes A, B, and C.")

print("⚙️ Connecting to Supabase PostgreSQL...")
try:
    pg_conn = psycopg2.connect(
        host="aws-1-ap-southeast-2.pooler.supabase.com", 
        port="5432", 
        database="postgres",
        user="postgres.vnunjidwtvevzaqamezi",
        password="mAJORPROJECT@1234"
    )
    pg_conn.autocommit = True
    print("✅ Connected to Supabase Postgres successfully!")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not connect to Supabase Postgres. Error: {e}")
    exit()

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
print("✅ Kafka Consumer Subscribed. Listening for live transactions...\n")
print("=" * 75)

def safe_hash(text, max_bins):
    text = str(text).strip().upper()
    hash_hex = hashlib.md5(text.encode('utf-8')).hexdigest()
    return int(hash_hex, 16) % max_bins

# ==========================================
# PHASE 2: MEMGRAPH INGESTION
# ==========================================
def ingest_live_transaction(payload):
    swift = payload['swift']
    trade = payload.get('trade', {})

    # FIX: Added fx_deviation_pct and applied_exchange_rate to database persistence
    query = """
    MERGE (sender:Account {id: $sender_id})
    MERGE (receiver:Account {id: $receiver_id})
    MERGE (t:Transaction {id: $msg_id})
    SET t.amount_usd = $amount, 
        t.target_currency = $target_currency,
        t.pattern_type = $pattern_type,
        t.fx_deviation_pct = $fx_dev,
        t.applied_exchange_rate = $ex_rate
    MERGE (sender)-[:SENDS]->(t)-[:TO]->(receiver)
    WITH t
    WHERE $doc_id IS NOT NULL
    MERGE (d:Document {id: $doc_id})
    SET d.commodity = $commodity, 
        d.price_deviation = $price_dev, 
        d.weight_gap_score = $weight_gap,
        d.qty = $qty,
        d.unit_price = $unit_price
    MERGE (t)-[:HAS_DOC]->(d)
    """
    
    # FIX: Prioritize target_currency over currency
    params = {
        "sender_id": swift['sender_id'], "receiver_id": swift['receiver_id'],
        "msg_id": swift['msg_id'], "amount": swift['amount'], 
        "target_currency": swift.get('target_currency', swift.get('currency', 'USD')),
        "pattern_type": swift.get('pattern_type', 'No_Pattern'),
        "fx_dev": swift.get('fx_deviation_pct', 0.0),
        "ex_rate": swift.get('applied_exchange_rate', 1.0),
        "doc_id": trade.get('doc_id'), "commodity": trade.get('commodity', 'UNKNOWN'),
        "price_dev": trade.get('price_deviation', 0.0), "weight_gap": trade.get('weight_gap_score', 0.0),
        "qty": trade.get('qty', 0.0), "unit_price": trade.get('unit_price', 0.0)
    }
    
    try:
        sender_bank = swift['sender_id'].split('_')[1]
        receiver_bank = swift['receiver_id'].split('_')[1]
    except IndexError:
        print(f"⚠️ DATA WARNING: Malformed IDs in transaction {swift['msg_id']}. Skipping ingestion.")
        return

    banks_to_update = set([sender_bank, receiver_bank])
    
    for bank_key in banks_to_update:
        driver = db_drivers.get(bank_key)
        if driver:
            with driver.session() as session:
                session.run(query, **params)

# ==========================================
# PHASE 3: DEEP SUBGRAPH EXTRACTION
# ==========================================
def get_deep_subgraph_tensors(tx_id, swift_payload, trade_payload):
    sender_id = swift_payload.get('sender_id')
    
    if not sender_id:
        return None, None, None, None, None, None
        
    try:
        sender_bank = sender_id.split('_')[1]
    except IndexError:
        return None, None, None, None, None, None
        
    driver = db_drivers.get(sender_bank)
    if not driver:
        return None, None, None, None, None, None

    query = """
    MATCH (root:Transaction {id: $tx_id})
    OPTIONAL MATCH (root)-[:HAS_DOC]->(trade:Document)
    WITH root, trade
    
    MATCH (root)-[r1]-(n1:Account)
    WITH root, trade, collect(r1) as edges1, collect(n1) as nodes1
    
    UNWIND nodes1 as n1
    OPTIONAL MATCH (n1)-[r2]-(n2:Transaction)
    WITH root, trade, edges1, nodes1, collect(r2)[..50] as edges2, collect(n2)[..50] as nodes2
    
    WITH root, trade, 
         nodes1 + [x IN nodes2 WHERE x IS NOT NULL] AS all_nodes,
         edges1 + [x IN edges2 WHERE x IS NOT NULL] AS all_edges
    
    UNWIND all_nodes AS n
    UNWIND all_edges AS r
    RETURN root, trade, collect(distinct n) AS nodes, collect(distinct r) AS edges
    """
    with driver.session() as session:
        result = session.run(query, tx_id=tx_id).single()
        
    if not result:
        raise ValueError(f"Graph not found for {tx_id}")

    root = result["root"]
    trade = result["trade"]
    
    all_nodes_dict = {root.element_id: root}
    if trade:
        all_nodes_dict[trade.element_id] = trade
        
    if result["nodes"]:
        for n in result["nodes"]:
            all_nodes_dict[n.element_id] = n
            
    nodes = list(all_nodes_dict.values())

    id_map = {n.element_id: n.get("id", "UNKNOWN") for n in nodes}
    ego_graph_json = {
        "total_nodes": len(nodes),
        "nodes": [{"id": n.get("id", "UNKNOWN"), "labels": list(n.labels)} for n in nodes],
        "edges": []
    }
    
    if result["edges"]:
        for edge in result["edges"]:
            ego_graph_json["edges"].append({
                "source": id_map.get(edge.start_node.element_id, "UNKNOWN"),
                "target": id_map.get(edge.end_node.element_id, "UNKNOWN"),
                "type": edge.type
            })

    node_to_idx = {}
    kyc_features = []
    ref_date = datetime(2026, 1, 1) 
    
    for i, node in enumerate(nodes):
        node_to_idx[node.element_id] = i
        if "Account" in node.labels:
            incorp_str = node.get("incorp_date", "2020-01-01")
            try:
                incorp_date = datetime.strptime(incorp_str, "%Y-%m-%d")
                age = float(max(0, (ref_date - incorp_date).days)) / 365.0
            except:
                age = 1.0 
                
            is_acct = 1.0
            shell = float(node.get("is_shell", 0.0))
            dorm_days = float(node.get("dorm_days", 0.0)) / 365.0
            entropy = float(node.get("device_entropy", 0.0))
            
            jur_id = safe_hash(node.get("jurisdiction", "UNKNOWN"), 50)
            ind_id = safe_hash(node.get("industry", "UNKNOWN"), 30)
            
            kyc = [is_acct, shell, dorm_days, entropy, age, jur_id, ind_id]
        else:
            kyc = [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]
        kyc_features.append(kyc)

    kyc_x = torch.tensor(kyc_features, dtype=torch.float32)

    source_nodes = []
    target_nodes = []
    if result["edges"]:
        for edge in result["edges"]:
            start_id = edge.start_node.element_id
            end_id = edge.end_node.element_id
            if start_id in node_to_idx and end_id in node_to_idx:
                source_nodes.append(node_to_idx[start_id])
                target_nodes.append(node_to_idx[end_id])
            
    if len(source_nodes) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # FIX: Align perfectly with target_currency fallback
    raw_amount = np.log1p(float(swift_payload.get('amount', 0.0)))
    fx_dev = float(swift_payload.get('fx_deviation_pct', 0.0)) / 100.0
    ex_rate = float(swift_payload.get('applied_exchange_rate', 1.0))
    curr_id = safe_hash(swift_payload.get('target_currency', swift_payload.get('currency', 'USD')), 20)
    pat_id = safe_hash(swift_payload.get('pattern_type', 'No_Pattern'), 10)

    seq_data = torch.tensor([[[raw_amount, fx_dev, ex_rate, curr_id, pat_id]]], dtype=torch.float32)
    swift_edge_attr = torch.tensor([[raw_amount]], dtype=torch.float32)

    if trade_payload:
        price_dev = float(trade_payload.get('price_deviation', 0.0)) / 100.0
        weight_gap = float(trade_payload.get('weight_gap_score', 0.0)) / 100.0
        qty = np.log1p(float(trade_payload.get('qty', 0.0)))
        unit_price = np.log1p(float(trade_payload.get('unit_price', 0.0)))
        com_id = safe_hash(trade_payload.get('commodity', 'UNKNOWN'), 30)
        
        trade_features = torch.tensor([[price_dev, weight_gap, qty, unit_price, com_id]], dtype=torch.float32)
    else:
        trade_features = torch.zeros((1, 5), dtype=torch.float32)

    return kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, ego_graph_json

# ==========================================
# PHASE 3.5: POSTGRES AUDIT LOGGER
# ==========================================
def log_verdict_to_postgres(tx_id, sender_id, receiver_id, amount, verdict_str, confidence, action, audit_metadata):
    query = """
    INSERT INTO aml_audit_log 
    (transaction_id, sender_account, receiver_account, amount, verdict, confidence_score, str_metadata, action_taken)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (transaction_id) DO NOTHING;
    """
    try:
        with pg_conn.cursor() as cur:
            json_payload = json.dumps(audit_metadata) if audit_metadata else None
            cur.execute(query, (tx_id, sender_id, receiver_id, amount, verdict_str, confidence, json_payload, action))
    except Exception as e:
        print(f"\n⚠️ WARNING: Failed to log transaction {tx_id} to Postgres: {e}")

# ==========================================
# PHASE 4: THE REAL-TIME PIPELINE LOOP
# ==========================================
def process_live_transaction(payload):
    swift = payload['swift']
    trade = payload.get('trade', {}) 
    tx_id = swift['msg_id']
    amount = float(swift['amount'])
    
    sender_id = swift.get('sender_id', 'UNKNOWN')
    receiver_id = swift.get('receiver_id', 'UNKNOWN')
    pattern = swift.get('pattern_type', 'STANDARD')
    
    if trade:
        unit_price = float(trade.get('unit_price', 1.0))
        market_avg = float(trade.get('market_avg', 1.0))
        declared_weight = float(trade.get('declared_weight_kg', 1.0))
        weight_gap_raw = float(trade.get('weight_gap_score', 0.0))

        trade['price_deviation'] = unit_price / market_avg if market_avg > 0 else 1.0
        trade['weight_gap_score'] = weight_gap_raw / declared_weight if declared_weight > 0 else 0.0

    ingest_live_transaction(payload)
        
    kyc_x, edge_index, swift_edge_attr, seq, trade_features, ego_graph_json = get_deep_subgraph_tensors(tx_id, swift, trade)
    
    if kyc_x is None:
        return

    with torch.no_grad():
        logits = model(kyc_x, edge_index, None, swift_edge_attr, seq, trade_features)
        
    probabilities = torch.softmax(logits, dim=1)[0]
    
    clean_prob = probabilities[0].item()
    watchlist_prob = probabilities[1].item()
    fraud_prob = probabilities[2].item()
    
    WATCHLIST_THRESHOLD = 0.15
    FRAUD_THRESHOLD = 0.25
    
    # FIX: Refactored if/elif/else block for clearer viva explanation
    if fraud_prob >= FRAUD_THRESHOLD and fraud_prob > clean_prob and fraud_prob > watchlist_prob:
        predicted_class = 2
        confidence = fraud_prob * 100
    elif watchlist_prob >= WATCHLIST_THRESHOLD and watchlist_prob > clean_prob:
        predicted_class = 1
        confidence = watchlist_prob * 100
    else:
        predicted_class = 0
        confidence = clean_prob * 100

    print(f"\n[INFERENCE REPORT] Transaction: {tx_id} | Type: {pattern} | Amount: ${amount:,.2f}")
    
    if predicted_class == 2:
        verdict = "FRAUD"
        action = "Funds Frozen. Routing to compliance team."
        print(f"🚨 VERDICT: [FRAUD] | Confidence: {confidence:.1f}%")
        print(f"   Action: {action}")
    elif predicted_class == 1:
        verdict = "WATCHLIST"
        action = "Transaction held for manual audit."
        print(f"⚠️ VERDICT: [WATCHLIST] | Confidence: {confidence:.1f}%")
        print(f"   Action: {action}")
    else:
        verdict = "CLEAN"
        action = "Cleared for settlement."
        print(f"✅ VERDICT: [CLEAN] | Confidence: {confidence:.1f}%")
        print(f"   Action: {action}")
    
    audit_metadata = {
        "pattern_detected": pattern,
        "trade_data": trade,
        "ego_graph": ego_graph_json
    }

    log_verdict_to_postgres(tx_id, sender_id, receiver_id, amount, verdict, confidence, action, audit_metadata)

try:
    for message in consumer:
        process_live_transaction(message.value)
except KeyboardInterrupt:
    print("\n🛑 Shutting down Inference Engine.")
finally:
    consumer.close()
    for driver in db_drivers.values():
        driver.close()
    if 'pg_conn' in locals() and pg_conn:
        pg_conn.close()
        print("✅ Postgres connection closed safely.")