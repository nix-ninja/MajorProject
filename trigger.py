import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer

# --- CONFIGURATION ---
KAFKA_TOPIC = 'incoming_swift_messages'

COMMODITY_PRICES = {
    'Textiles': 50.0, 'Electronics': 450.0, 'Scrap Metal': 120.0, 
    'Luxury Goods': 1200.0, 'Grain': 30.0
}
# Realistic Weighted Currency Distribution
CURRENCIES = ['USD', 'EUR', 'GBP', 'SGD', 'AED']
CURRENCY_WEIGHTS = [0.55, 0.20, 0.10, 0.10, 0.05]

# --- EXCHANGE RATE MOCKER ---
EXCHANGE_RATES = {'USD': 1.0, 'EUR': 1.08, 'GBP': 1.25, 'SGD': 0.74, 'AED': 0.27}

# 1 & 3: FEDERATED UNIVERSE SETUP
BANKS = ['A', 'B', 'C']
KNOWN_ACCOUNTS = [f"ACC_{bank}_{i:06d}" for bank in BANKS for i in range(1, 75000, 10)]

# 1 & 4: PERSISTENT SHELL POOL & COMMODITY AFFINITY 
SHELL_REGISTRY = {}
for i in range(77000, 77050):
    bank_prefix = random.choice(BANKS)
    shell_id = f"ACC_{bank_prefix}_{i:06d}"
    SHELL_REGISTRY[shell_id] = {
        'affinity': random.choice(list(COMMODITY_PRICES.keys()))
    }

# --- HELPER FUNCTION FOR FRAUD TOPOLOGIES ---
def create_fraud_payload(sender, receiver, qty, unit_price, market_avg, commodity, pattern_type):
    msg_id = f"SW_LIVE_FRAUD_{str(uuid.uuid4())[:8].upper()}"
    doc_id = f"DOC_LIVE_{str(uuid.uuid4())[:8].upper()}"
    
    declared_weight_kg = qty * 2.2
    actual_weight_kg = declared_weight_kg * random.uniform(0.75, 0.90) 
    amount = round(qty * unit_price, 2)
    
    tx_currency = random.choices(CURRENCIES, weights=CURRENCY_WEIGHTS, k=1)[0]
    
    ex_rate = EXCHANGE_RATES[tx_currency]
    fx_dev = round(random.uniform(2.5, 15.0), 2) # High FX deviation
    
    payload = {
        "swift": {
            "msg_id": msg_id, "sender_id": sender, "receiver_id": receiver,
            "amount": amount, "currency": tx_currency, "timestamp": datetime.now().isoformat(),
            "pattern_type": pattern_type,
            "fx_deviation_pct": fx_dev,              
            "applied_exchange_rate": ex_rate         
        },
        "trade": {
            "doc_id": doc_id, "commodity": commodity, "qty": qty,
            "unit_price": round(unit_price, 2), "market_avg": market_avg,
            "price_deviation": round(unit_price - market_avg, 2),
            "declared_weight_kg": round(declared_weight_kg, 2),
            "actual_weight_kg": round(actual_weight_kg, 2),
            "weight_gap_score": round(declared_weight_kg - actual_weight_kg, 2)
        }
    }
    return (payload, "🚨 FRAUD")

print("⚙️ Initializing Advanced Federated Kafka Producer...")
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print("🚀 Starting Advanced Synthetic Live Stream...\n")
print("-" * 75)

burst_remaining = 0

try:
    while True:
        rand_roll = random.random()
        if rand_roll < 0.05:
            tx_class = 2  
        elif rand_roll < 0.15:
            tx_class = 1  
        else:
            tx_class = 0  

        payloads_to_send = []

        if tx_class == 2:
            # ==========================================
            # EXACT VOCABULARY AND TOPOLOGIES
            # ==========================================
            pattern_type = random.choice(['FAN-IN', 'FAN-OUT', 'BIPARTITE', 'STACK', 'CYCLE', 'GATHER-SCATTER']) 
            
            print(f"\n🚨 [ORCHESTRATING FRAUD RING] Generating Cross-Bank {pattern_type} topology...")

            if pattern_type == 'FAN-IN':
                target_account = random.choice(KNOWN_ACCOUNTS)
                smurfs = random.sample(list(SHELL_REGISTRY.keys()), random.randint(3, 5))
                for smurf in smurfs:
                    commodity = SHELL_REGISTRY[smurf]['affinity']
                    market_avg = COMMODITY_PRICES[commodity]
                    qty = random.randint(500, 1500)
                    unit_price = market_avg * random.uniform(1.20, 1.80) 
                    payloads_to_send.append(
                        create_fraud_payload(smurf, target_account, qty, unit_price, market_avg, commodity, pattern_type)
                    )

            elif pattern_type == 'FAN-OUT':
                source_account = random.choice(list(SHELL_REGISTRY.keys()))
                commodity = SHELL_REGISTRY[source_account]['affinity']
                market_avg = COMMODITY_PRICES[commodity]
                receivers = random.sample(KNOWN_ACCOUNTS, random.randint(3, 5))
                for rec in receivers:
                    qty = random.randint(500, 1500)
                    unit_price = market_avg * random.uniform(1.20, 1.80)
                    payloads_to_send.append(
                        create_fraud_payload(source_account, rec, qty, unit_price, market_avg, commodity, pattern_type)
                    )

            elif pattern_type == 'CYCLE':
                acc_A = random.choice(list(SHELL_REGISTRY.keys()))
                acc_B = random.choice(KNOWN_ACCOUNTS)
                acc_C = random.choice(list(SHELL_REGISTRY.keys()))
                commodity = SHELL_REGISTRY[acc_A]['affinity']
                market_avg = COMMODITY_PRICES[commodity]
                cycle_path = [(acc_A, acc_B), (acc_B, acc_C), (acc_C, acc_A)]
                for sender, receiver in cycle_path:
                    qty = random.randint(1000, 2000)
                    unit_price = market_avg * random.uniform(1.50, 2.00) 
                    payloads_to_send.append(
                        create_fraud_payload(sender, receiver, qty, unit_price, market_avg, commodity, pattern_type)
                    )

            elif pattern_type == 'BIPARTITE':
                # Two sets of nodes transacting exclusively with each other
                senders = random.sample(list(SHELL_REGISTRY.keys()), 2)
                receivers = random.sample(KNOWN_ACCOUNTS, 2)
                commodity = SHELL_REGISTRY[senders[0]]['affinity']
                market_avg = COMMODITY_PRICES[commodity]
                for s in senders:
                    for r in receivers:
                        qty = random.randint(500, 1000)
                        unit_price = market_avg * random.uniform(1.30, 1.70)
                        payloads_to_send.append(
                            create_fraud_payload(s, r, qty, unit_price, market_avg, commodity, pattern_type)
                        )

            elif pattern_type == 'STACK':
                # A linear chain of transactions moving money deep into the network
                node1 = random.choice(list(SHELL_REGISTRY.keys()))
                node2 = random.choice(KNOWN_ACCOUNTS)
                node3 = random.choice(KNOWN_ACCOUNTS)
                node4 = random.choice(list(SHELL_REGISTRY.keys()))
                commodity = SHELL_REGISTRY[node1]['affinity']
                market_avg = COMMODITY_PRICES[commodity]
                
                stack_path = [(node1, node2), (node2, node3), (node3, node4)]
                for s, r in stack_path:
                    qty = random.randint(800, 1200)
                    unit_price = market_avg * random.uniform(1.20, 1.60)
                    payloads_to_send.append(
                        create_fraud_payload(s, r, qty, unit_price, market_avg, commodity, pattern_type)
                    )

            elif pattern_type == 'GATHER-SCATTER':
                # Multiple senders pool into one node, which immediately splits it out
                gatherers = random.sample(list(SHELL_REGISTRY.keys()), 3)
                middleman = random.choice(KNOWN_ACCOUNTS)
                scatterers = random.sample(KNOWN_ACCOUNTS, 3)
                
                commodity = SHELL_REGISTRY[gatherers[0]]['affinity']
                market_avg = COMMODITY_PRICES[commodity]
                
                for g in gatherers:
                    qty = random.randint(400, 700)
                    unit_price = market_avg * random.uniform(1.10, 1.40)
                    payloads_to_send.append(
                        create_fraud_payload(g, middleman, qty, unit_price, market_avg, commodity, pattern_type)
                    )
                for s in scatterers:
                    qty = random.randint(300, 600)
                    unit_price = market_avg * random.uniform(1.10, 1.40)
                    payloads_to_send.append(
                        create_fraud_payload(middleman, s, qty, unit_price, market_avg, commodity, pattern_type)
                    )

        else:
            # ==========================================
            # STANDARD ISOLATED TRANSACTION
            # ==========================================
            msg_id = f"SW_LIVE_{str(uuid.uuid4())[:8].upper()}"
            doc_id = f"DOC_LIVE_{str(uuid.uuid4())[:8].upper()}"
            
            sender = random.choice(KNOWN_ACCOUNTS)
            receiver = random.choice([acc for acc in KNOWN_ACCOUNTS if acc != sender])
            
            commodity = random.choice(list(COMMODITY_PRICES.keys()))
            market_avg = COMMODITY_PRICES[commodity]
            qty = random.randint(10, 200)
            declared_weight_kg = qty * 2.2
            tx_currency = random.choices(CURRENCIES, weights=CURRENCY_WEIGHTS, k=1)[0]
            ex_rate = EXCHANGE_RATES[tx_currency]
            
            if tx_class == 1:
                unit_price = market_avg * random.uniform(1.10, 1.30)
                actual_weight_kg = declared_weight_kg * random.uniform(0.88, 0.95)
                pattern = "No_Pattern" 
                fx_dev = round(random.uniform(1.0, 3.5), 2) 
                status = "⚠️ WATCH "
            else:
                unit_price = market_avg * random.uniform(0.95, 1.05)
                actual_weight_kg = declared_weight_kg * random.uniform(0.98, 1.0)
                pattern = "No_Pattern" 
                fx_dev = round(random.uniform(0.1, 0.9), 2) 
                status = "✅ LEGIT "
                
            amount = round(qty * unit_price, 2)
            
            payload = {
                "swift": {
                    "msg_id": msg_id, "sender_id": sender, "receiver_id": receiver,
                    "amount": amount, "currency": tx_currency, "timestamp": datetime.now().isoformat(),
                    "pattern_type": pattern,
                    "fx_deviation_pct": fx_dev,              
                    "applied_exchange_rate": ex_rate         
                },
                "trade": {
                    "doc_id": doc_id, "commodity": commodity, "qty": qty,
                    "unit_price": round(unit_price, 2), "market_avg": market_avg,
                    "price_deviation": round(unit_price - market_avg, 2),
                    "declared_weight_kg": round(declared_weight_kg, 2),
                    "actual_weight_kg": round(actual_weight_kg, 2),
                    "weight_gap_score": round(declared_weight_kg - actual_weight_kg, 2)
                }
            }
            payloads_to_send.append((payload, status))

        # --- SEND ALL GENERATED PAYLOADS ---
        for payload, status in payloads_to_send:
            producer.send(KAFKA_TOPIC, value=payload)
            swift_data = payload['swift']
            print(f"[{time.strftime('%H:%M:%S')}] {status} | {swift_data['msg_id']} | {swift_data['sender_id']} -> {swift_data['receiver_id']} | ${swift_data['amount']:,.2f}")
            
        # ==========================================
        # BURST LATENCY ENGINE
        # ==========================================
        if random.random() < 0.02 and burst_remaining <= 0:
            burst_remaining = random.randint(10, 25)
            print("\n🌩️ [BURST MODE ACTIVATED] Initiating rapid transaction sequence...")

        if burst_remaining > 0:
            time.sleep(random.uniform(0.01, 0.1))
            burst_remaining -= 1
            if burst_remaining == 0:
                print("🔇 [QUIET PERIOD] Burst complete. System entering silent phase...")
                time.sleep(random.uniform(6.0, 12.0)) 
        else:
            time.sleep(random.uniform(1.0, 2.5))

except KeyboardInterrupt:
    print("\n🛑 Stopping advanced federated synthetic stream.")
finally:
    producer.close()