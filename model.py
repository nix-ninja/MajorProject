import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class TBML_DetectionModel(nn.Module):
    def __init__(self, hidden_dim=64, lstm_hidden=64, classes=3):
        super(TBML_DetectionModel, self).__init__()
        
        # ---------------------------------------------------------
        # 1. EMBEDDINGS (Categorical IDs -> Dense Vectors)
        # ---------------------------------------------------------
        self.jurisdiction_emb = nn.Embedding(num_embeddings=50, embedding_dim=8)
        self.industry_emb = nn.Embedding(num_embeddings=30, embedding_dim=8)
        self.currency_emb = nn.Embedding(num_embeddings=20, embedding_dim=4)
        self.pattern_emb = nn.Embedding(num_embeddings=10, embedding_dim=4)
        self.commodity_emb = nn.Embedding(num_embeddings=30, embedding_dim=8)

        # ---------------------------------------------------------
        # 2. THE DEEP GRAPH BRANCH (KYC + Network Topology)
        # Input: 5 Numericals (is_acct, shell, dorm, entropy, age) + 8 (Jur) + 8 (Ind) = 21 dims
        # ---------------------------------------------------------
        self.gat1 = GATv2Conv(in_channels=21, out_channels=hidden_dim, heads=4, concat=False, dropout=0.6)
        self.gat2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, heads=4, concat=False, dropout=0.6)
        self.gat3 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, heads=4, concat=False, dropout=0.6)

        # ---------------------------------------------------------
        # 3. THE TEMPORAL BRANCH (SWIFT Sequences)
        # Input: Graph Context (64) + 3 Numericals (amt, fx, rate) + 4 (Curr) + 4 (Pat) = 75 dims
        # ---------------------------------------------------------
        self.lstm = nn.LSTM(input_size=75, hidden_size=lstm_hidden, batch_first=True)

        # ---------------------------------------------------------
        # 4. THE TRADE BRANCH
        # Input: 4 Numericals (price_dev, weight_gap, qty, price) + 8 (Com) = 12 dims
        # ---------------------------------------------------------
        self.trade_mlp = nn.Sequential(
            nn.Linear(12, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 5. FUSION & CLASSIFICATION
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout for federated stability
            nn.Linear(64, classes)
        )

    def forward(self, kyc_x, edge_index, batch_idx, swift_edge_attr, seq_data, trade_features):
        
        # ==========================================
        # STEP 1: PARSE AND EMBED KYC (Nodes)
        # ==========================================
        # Extract the continuous floats vs the categorical IDs
        kyc_nums = kyc_x[:, :5]                # First 5 are floats
        jur_ids = kyc_x[:, 5].long()           # Cast to integer for embedding lookup
        ind_ids = kyc_x[:, 6].long()           
        
        # Convert IDs to rich mathematical vectors
        j_vec = self.jurisdiction_emb(jur_ids)
        i_vec = self.industry_emb(ind_ids)
        
        # Re-concatenate into a unified 21-dim vector for the GAT
        kyc_combined = torch.cat([kyc_nums, j_vec, i_vec], dim=1) 

        # ==========================================
        # STEP 2: GRAPH MESSAGE PASSING
        # ==========================================
        x = F.dropout(F.relu(self.gat1(kyc_combined, edge_index)), p=0.2, training=self.training)
        x = F.dropout(F.relu(self.gat2(x, edge_index)), p=0.2, training=self.training)
        enriched = F.dropout(F.relu(self.gat3(x, edge_index)), p=0.2, training=self.training)

        # Handle PyG batching safely
        if batch_idx is None:
            batch_idx = torch.zeros(enriched.size(0), dtype=torch.long, device=enriched.device)
            
        graph_ctx = global_mean_pool(enriched, batch_idx).unsqueeze(1)

        # ==========================================
        # STEP 3: PARSE AND EMBED SWIFT (Sequence)
        # ==========================================
        if seq_data.dim() == 4:
            seq_data = seq_data.squeeze(0) 
            
        swift_nums = seq_data[:, :, :3]
        curr_ids = seq_data[:, :, 3].long()
        pat_ids = seq_data[:, :, 4].long()
        
        c_vec = self.currency_emb(curr_ids)
        p_vec = self.pattern_emb(pat_ids)
        
        # Inject the Graph Context directly into every step of the LSTM sequence
        graph_expanded = graph_ctx.expand(seq_data.size(0), seq_data.size(1), -1)
        combined_seq = torch.cat([swift_nums, c_vec, p_vec, graph_expanded], dim=-1)

        lstm_out, _ = self.lstm(combined_seq)
        lstm_final = F.dropout(lstm_out[:, -1, :], p=0.3, training=self.training)
        
        # ==========================================
        # STEP 4: PARSE AND EMBED TRADE DOCUMENTS
        # ==========================================
        if trade_features.dim() == 3: 
             trade_features = trade_features.squeeze(1)
             
        trade_nums = trade_features[:, :4]
        com_ids = trade_features[:, 4].long()
        
        com_vec = self.commodity_emb(com_ids)
        trade_combined = torch.cat([trade_nums, com_vec], dim=1)
        
        trade_emb = self.trade_mlp(trade_combined)
        
        # ==========================================
        # STEP 5: FINAL FUSION & CLASSIFICATION
        # ==========================================
        fused = torch.cat([lstm_final, trade_emb], dim=1)
        return self.classifier(fused)