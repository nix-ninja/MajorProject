import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from neo4j import GraphDatabase
import argparse
from collections import OrderedDict
from sklearn.metrics import average_precision_score, precision_recall_curve

from model import TBML_DetectionModel
from collections import Counter
import random
import numpy as np
import hashlib
from datetime import datetime

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  
        self.gamma = gamma  

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- IMPORTS FOR GRAPHING & METRICS ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

# --- 1. CONFIG & DOCKER PORT MAPPING ---
parser = argparse.ArgumentParser()
parser.add_argument("--bank", type=str, required=True, choices=['banka', 'bankb', 'bankc'])
args = parser.parse_args()

PORT_MAP = {
    "banka": 4000,
    "bankb": 4001,
    "bankc": 4002
}
MEMGRAPH_URI = f"bolt://localhost:{PORT_MAP[args.bank]}"
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""


# --- 2. GRAPH GENERATOR HELPER FUNCTION ---
def generate_presentation_metrics(y_true, y_pred, y_probs, bank_name):
    classes = [0, 1, 2]
    class_names = ['Clean (0)', 'Watchlist (1)', 'Fraud (2)']
    
    print(f"\n--- {bank_name.upper()} FINAL EVALUATION METRICS ---")
    print(classification_report(y_true, y_pred, labels=classes, target_names=class_names, zero_division=0))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{bank_name.upper()} - TBML Confusion Matrix')
    plt.ylabel('Actual Truth')
    plt.xlabel('AI Prediction')
    plt.tight_layout()
    plt.savefig(f'{bank_name}_confusion_matrix_deep.png', dpi=300)
    plt.close()

    y_true_bin = label_binarize(y_true, classes=classes)
    fraud_class_idx = 2
    
    # --- PR AUC FOR FRAUD CLASS ---
    if np.sum(y_true_bin[:, fraud_class_idx]) > 0:
        pr_auc = average_precision_score(
            y_true_bin[:, fraud_class_idx],
            y_probs[:, fraud_class_idx]
        )
        print(f"Fraud PR-AUC: {pr_auc:.4f}")
    
        fpr, tpr, _ = roc_curve(y_true_bin[:, fraud_class_idx], y_probs[:, fraud_class_idx])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Fraud ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{bank_name.upper()} - Receiver Operating Characteristic (Fraud)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'{bank_name}_roc_curve_deep.png', dpi=300)
        plt.close()
    
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true_bin[:, fraud_class_idx],
            y_probs[:, fraud_class_idx]
        )

        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='purple', lw=2, label=f'Fraud PR Curve (AP = {pr_auc:.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{bank_name.upper()} - Precision Recall Curve (Fraud)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"{bank_name}_pr_curve_deep.png", dpi=300)
        plt.close()

    # 3. Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    
    x = np.arange(len(class_names))
    width = 0.25 

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    rects2 = ax.bar(x, recall, width, label='Recall', color='lightgreen')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='salmon')

    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_title(f'{bank_name.upper()} - Model Performance by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim([0.0, 1.1]) 
    ax.legend(loc='upper right')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(f'{bank_name}_performance_metrics_deep.png', dpi=300)
    plt.close()


# --- 3. MEMGRAPH DATA BRIDGE ---

def get_entity_temporal_split_ids(uri, user, password, split_ratio=0.8):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        print("Fetching transactions with labels for stratified temporal splitting...")
        query = """
        MATCH (t:Transaction)
        WHERE t.timestamp IS NOT NULL AND t.label IS NOT NULL
        RETURN t.id AS tx_id, t.timestamp AS ts, t.label AS label
        ORDER BY ts ASC
        """
        result = session.run(query)
        
        by_label = {0: [], 1: [], 2: []}
        for r in result:
            lbl = int(r["label"])
            if lbl in by_label:
                by_label[lbl].append({"tx_id": r["tx_id"], "ts": r["ts"]})
                
    driver.close()
    
    train_ids = []
    test_ids = []
    
    for lbl, txs in by_label.items():
        if len(txs) == 0:
            continue
            
        split_idx = int(len(txs) * split_ratio)
        class_train = [tx["tx_id"] for tx in txs[:split_idx]]
        class_test = [tx["tx_id"] for tx in txs[split_idx:]]
        
        train_ids.extend(class_train)
        test_ids.extend(class_test)
        
    print("\n=== STRATIFIED TEMPORAL SPLIT COMPLETE ===")
    print(f"Total Train Available across all classes: {len(train_ids)}")
    print(f"Total Test Available across all classes: {len(test_ids)}")
    
    np.random.seed(42)
    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)
    
    print("\n=== REDUCED BALANCED DATASET ===")
    print(f"Final Train Transactions for Model : {len(train_ids)}")
    print(f"Final Test Transactions for Model  : {len(test_ids)}")
    
    return train_ids, test_ids

class MemgraphTBMLDataset(Dataset):
    def __init__(self, uri, user, password, transaction_ids):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.transaction_ids = transaction_ids

    def __len__(self):
        return len(self.transaction_ids)
        
    def safe_hash(self, text, max_bins):
        text = str(text).strip().upper()
        hash_hex = hashlib.md5(text.encode('utf-8')).hexdigest()
        return int(hash_hex, 16) % max_bins

    def __getitem__(self, idx):
        target_msg_id = self.transaction_ids[idx]
        
        query = """
        MATCH (root:Transaction {id: $target_msg_id})
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
        
        with self.driver.session() as session:
            result = session.run(query, target_msg_id=target_msg_id).single()

        root = result["root"]
        trade = result["trade"]
        
        all_nodes_dict = {root.element_id: root}
        if trade:
            all_nodes_dict[trade.element_id] = trade
            
        for n in result["nodes"]:
            all_nodes_dict[n.element_id] = n
            
        nodes = list(all_nodes_dict.values()) 

        # --- 1. UNIVERSAL NODE VECTOR (KYC) ---
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
                
                jur_id = self.safe_hash(node.get("jurisdiction", "UNKNOWN"), 50)
                ind_id = self.safe_hash(node.get("industry", "UNKNOWN"), 30)
                
                kyc = [is_acct, shell, dorm_days, entropy, age, jur_id, ind_id]
            else:
                kyc = [0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]
                
            kyc_features.append(kyc)

        kyc_x = torch.tensor(kyc_features, dtype=torch.float32)

        source_nodes = []
        target_nodes = []
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

        # --- 2. SWIFT / TEMPORAL ---
        raw_amount = np.log1p(float(root.get("amount_usd", 0.0)))
        fx_dev = float(root.get("fx_deviation_pct", 0.0)) / 100.0
        ex_rate = float(root.get("applied_exchange_rate", 1.0))
        curr_id = self.safe_hash(root.get("target_currency", "USD"), 20)
        pat_id = self.safe_hash(root.get("pattern_type", "No_Pattern"), 10)
        
        seq_data = torch.tensor([[[raw_amount, fx_dev, ex_rate, curr_id, pat_id]]], dtype=torch.float32)
        swift_edge_attr = torch.tensor([[raw_amount]], dtype=torch.float32)

        # --- 3. TRADE DOCUMENTS ---
        if trade:
            price_dev = float(trade.get("price_deviation", 0.0)) / 100.0
            weight_gap = float(trade.get("weight_gap_score", 0.0)) / 100.0
            qty = np.log1p(float(trade.get("qty", 0.0)))
            unit_price = np.log1p(float(trade.get("unit_price", 0.0)))
            com_id = self.safe_hash(trade.get("commodity", "UNKNOWN"), 30)
            
            trade_features = torch.tensor([[price_dev, weight_gap, qty, unit_price, com_id]], dtype=torch.float32)
        else:
            trade_features = torch.zeros((1, 5), dtype=torch.float32)

        label = torch.tensor(int(root.get("label", 0)), dtype=torch.long)

        return kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label


# --- 4. FLOWER FEDERATED CLIENT ---
class BankClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, class_weights):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = FocalLoss(alpha=class_weights, gamma=1.2)
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        
        self.round_counter = 0
        self.history = {'round': [], 'accuracy': [], 'precision': [], 'recall': []}

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        try:
            import sys
            print(f"\n--- ENTERING FIT ---")
            sys.stdout.flush()
            
            self.set_parameters(parameters)
            print("1. Parameters set successfully.")
            sys.stdout.flush()
            
            self.model.train()
            print("2. Starting training loop...")
            sys.stdout.flush()
            
            # Trained over 3 Epochs to properly calibrate probabilities
            for epoch in range(3): 
                for kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label in self.train_loader:
                    kyc_x, edge_index = kyc_x[0], edge_index[0]
                    swift_edge_attr, label = swift_edge_attr[0], label[0]
                    trade_features = trade_features[0].unsqueeze(0)

                    self.optimizer.zero_grad()
                    logits = self.model(kyc_x, edge_index, None, swift_edge_attr, seq_data, trade_features)
                    
                    loss = self.criterion(logits, label.unsqueeze(0))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            
            print("3. Training loop complete.")
            sys.stdout.flush()
            
            if hasattr(self, 'scheduler'):
                self.scheduler.step()  
                
            return self.get_parameters(config={}), len(self.train_loader.dataset), {}
            
        except Exception as e:
            import traceback
            import sys
            err = traceback.format_exc()
            print(f"\n❌ FATAL CRASH IN FIT():\n{err}")
            sys.stdout.flush()
            raise e

    def evaluate(self, parameters, config):
        try:
            import sys
            print(f"\n--- ENTERING EVALUATE ---")
            sys.stdout.flush()
            
            self.set_parameters(parameters)
            self.model.eval()
            
            total_loss = 0.0
            all_labels = []
            all_probs = [] 

            print("2. Processing test batches...")
            sys.stdout.flush()
            
            with torch.no_grad():
                for kyc_x, edge_index, swift_edge_attr, seq_data, trade_features, label in self.test_loader:
                    kyc_x, edge_index = kyc_x[0], edge_index[0]
                    swift_edge_attr, label = swift_edge_attr[0], label[0]
                    trade_features = trade_features[0].unsqueeze(0)

                    logits = self.model(kyc_x, edge_index, None, swift_edge_attr, seq_data, trade_features)
                    loss = self.criterion(logits, label.unsqueeze(0))
                    
                    total_loss += loss.item()
                    probs = F.softmax(logits, dim=1)
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.append(label.cpu().item())

            if len(all_labels) == 0:
                return 0.0, 0, {"accuracy": 0.0, "f1_macro": 0.0}

            y_true = np.array(all_labels)
            y_probs = np.array(all_probs)
            
            clean_probs = y_probs[:, 0]
            watchlist_probs = y_probs[:, 1]
            fraud_probs = y_probs[:, 2]

            # --- STATIC THRESHOLDS & DOMINANCE RULE ---
            print("3. Applying Static Thresholds and Dominance Logic...")
            sys.stdout.flush()
            
            WATCHLIST_THRESHOLD = 0.15
            FRAUD_THRESHOLD = 0.25
            
            y_pred = np.zeros(len(y_probs), dtype=int)
            
            # Watchlist Rule
            watchlist_mask = (watchlist_probs >= WATCHLIST_THRESHOLD) & (watchlist_probs > clean_probs)
            y_pred[watchlist_mask] = 1
            
            # Fraud Dominance Rule
            fraud_mask = (fraud_probs >= FRAUD_THRESHOLD) & (fraud_probs > clean_probs) & (fraud_probs > watchlist_probs)
            y_pred[fraud_mask] = 2
            
            all_preds = y_pred.tolist()

            print("5. Generating presentation metrics and graphs...")
            sys.stdout.flush()
            
            generate_presentation_metrics(y_true, y_pred, y_probs, args.bank)

            accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            from sklearn.metrics import precision_score, recall_score
            macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            macro_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            self.round_counter += 1
            self.history['round'].append(self.round_counter)
            self.history['accuracy'].append(accuracy)
            self.history['precision'].append(macro_prec)
            self.history['recall'].append(macro_rec)

            plt.figure(figsize=(8, 5))
            plt.plot(self.history['round'], self.history['accuracy'], marker='o', color='royalblue', linewidth=2)
            plt.title(f'{args.bank.upper()} - Accuracy Over Time')
            plt.xlabel('Federated Round')
            plt.ylabel('Accuracy')
            plt.xticks(self.history['round'])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f'{args.bank}_progression_accuracy.png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(self.history['round'], self.history['precision'], marker='s', color='seagreen', linewidth=2)
            plt.title(f'{args.bank.upper()} - Macro Precision Over Time')
            plt.xlabel('Federated Round')
            plt.ylabel('Precision')
            plt.xticks(self.history['round'])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f'{args.bank}_progression_precision.png', dpi=300)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(self.history['round'], self.history['recall'], marker='^', color='darkorange', linewidth=2)
            plt.title(f'{args.bank.upper()} - Macro Recall Over Time')
            plt.xlabel('Federated Round')
            plt.ylabel('Recall')
            plt.xticks(self.history['round'])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f'{args.bank}_progression_recall.png', dpi=300)
            plt.close()

            print(f"Accuracy     : {accuracy:.4f}")
            print(f"Macro F1     : {macro_f1:.4f}")
            sys.stdout.flush()

            safe_loss = float(total_loss / len(self.test_loader)) if len(self.test_loader) > 0 else 0.0
            
            print("6. Evaluation successful! Returning to server.")
            sys.stdout.flush()
            return safe_loss, len(self.test_loader.dataset), {"accuracy": float(accuracy), "f1_macro": float(macro_f1)}
            
        except Exception as e:
            import traceback
            import sys
            err = traceback.format_exc()
            print(f"\n❌ FATAL CRASH IN EVALUATE():\n{err}")
            sys.stdout.flush()
            raise e

if __name__ == "__main__":
    print(f"🏦 Initializing {args.bank.upper()} on port {PORT_MAP[args.bank]}...")

    train_ids, test_ids = get_entity_temporal_split_ids(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS)
    train_dataset = MemgraphTBMLDataset(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS, train_ids)
    test_dataset = MemgraphTBMLDataset(MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASS, test_ids)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Re-instantiate the model to match the architecture in model.py
    model = TBML_DetectionModel(hidden_dim=64, lstm_hidden=64, classes=3)

    labels = []
    for _, _, _, _, _, label in train_dataset:
        labels.append(label.item())

    counts = Counter(labels)
    print("Class Counts:", counts)

    # --- MODERATED WEIGHTS ---
    class_weights = torch.tensor([1.0, 4.0, 8.0], dtype=torch.float32)
    print("🚨 Using Moderated Class Weights:", class_weights)

    client = BankClient(model, train_loader, test_loader, class_weights)
    
    print(f"🚀 {args.bank.upper()} connecting to Central Server...")
    
    fl.client.start_client(
        server_address="127.0.0.1:8085", 
        client=client.to_client()
    )