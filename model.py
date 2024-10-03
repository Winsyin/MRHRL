import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

EPS = torch.tensor(1e-15)

class Hgnn(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(Hgnn, self).__init__()
        self.conv1 = HypergraphConv(in_channels, dim_1)
        self.conv2 = HypergraphConv(dim_1, dim_1 // 4)

    def forward(self, x, edge):
        x = torch.relu(self.conv1(x, edge))
        x = torch.relu(self.conv2(x, edge))
        return x

class HgnnEncoder(nn.Module):
    def __init__(self, in_channels, dim_1):
        super(HgnnEncoder, self).__init__()


        self.conv = Hgnn(in_channels, dim_1)

        self.attn_c = FusionAttention(dim_1//4)
        self.attn_m = FusionAttention(dim_1//4)


    def generate_hye_emb(self, node_embedding, hye_len, hye_node):
        zeros = torch.zeros(1, list(node_embedding.shape)[1])
        node_embedding = torch.cat([zeros, node_embedding], 0)
        seq_h = node_embedding[hye_node]

        hs = torch.div(torch.sum(seq_h, 1), hye_len + EPS)
        return hs


    def forward(self, x_ls, hyg_ls):


        x1_cm = self.conv(x_ls[0], hyg_ls[0][0])
        x1_mc = self.conv(x_ls[1], hyg_ls[1][0])
        x1_cc = self.conv(x_ls[2], hyg_ls[2][0])
        x1_mm = self.conv(x_ls[3], hyg_ls[3][0])



        x2_cm = self.generate_hye_emb(x1_cm, hyg_ls[0][1], hyg_ls[0][2])
        x2_mc = self.generate_hye_emb(x1_mc, hyg_ls[1][1], hyg_ls[1][2])
        x2_cc = self.generate_hye_emb(x1_cc, hyg_ls[2][1], hyg_ls[2][2])
        x2_mm = self.generate_hye_emb(x1_mm, hyg_ls[3][1], hyg_ls[3][2])



        x_c = torch.stack((x1_mc, x2_cm, x1_cc, x2_cc), dim=1)
        x_m = torch.stack((x1_cm, x2_mc, x1_mm, x2_mm), dim=1)


        h_c = self.attn_c(x_c)
        h_m = self.attn_m(x_m)

        return h_c, h_m



class BioEncoder(nn.Module):
    def __init__(self, dim_circ, dim_mi, output):
        super(BioEncoder, self).__init__()
        self.circ_layer1 = nn.Linear(dim_circ, output)
        self.batch_circ1 = nn.BatchNorm1d(output)

        self.mi_layer1 = nn.Linear(dim_mi, output)
        self.batch_mi1 = nn.BatchNorm1d(output)
        self.relu = nn.ReLU()
        self.reset_para()


    def reset_para(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, circRNA_feature, miRNA_feature):
        x_circ = self.circ_layer1(circRNA_feature)
        x_circ = self.batch_circ1(F.relu(x_circ))

        x_mi = self.mi_layer1(miRNA_feature)
        x_mi = self.batch_mi1(F.relu(x_mi))

        return x_circ, x_mi



class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc23 = nn.Linear(in_channels // 4, in_channels // 8)
        self.batch23 = nn.BatchNorm1d(in_channels // 8)
        self.fc3 = nn.Linear(in_channels // 8, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, circ_embed, mi_embed, circRNA_id, miRNA_id):
        h_0 = torch.cat((circ_embed[circRNA_id, :], mi_embed[miRNA_id, :]), 1)
        h_1 = torch.tanh(self.fc1(h_0))
        h_1 = self.batch1(h_1)
        h_2 = torch.tanh(self.fc2(h_1))
        h_2 = self.batch2(h_2)
        h_23 = torch.tanh(self.fc23(h_2))
        h_23 = self.batch23(h_23)
        h_3 = self.fc3(h_23)
        return h_23, torch.sigmoid(h_3.squeeze(dim=1))

class FusionAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(FusionAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0], ) + beta.shape)

        return (beta * z).sum(1)




class MRHRL(torch.nn.Module):
    def __init__(self, bio_encoder, hygraph_encoder, decoder):
        super(MRHRL, self).__init__()
        self.bio_encoder = bio_encoder
        self.hygraph_encoder = hygraph_encoder
        self.decoder = decoder

    def forward(self, circ_feature, mi_feature, circ_id, mi_id, hyg_ls):

        x_circ, x_mi = self.bio_encoder(circ_feature, mi_feature)
        x_ls = [x_mi, x_circ, x_circ, x_mi]
        hc, hm = self.hygraph_encoder(x_ls, hyg_ls)

        out1 = F.normalize(hc, dim=1, p=2)
        out2 = F.normalize(hm, dim=1, p=2)

        emb, res = self.decoder(out1, out2, circ_id, mi_id)


        return emb, res, out1, out2
