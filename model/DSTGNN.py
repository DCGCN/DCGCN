import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class DSTGNN(nn.Module):
    def __init__(self, SE, infea, outfea, k, d, L):
        super(DSTGNN, self).__init__()
        self.Encoder = nn.ModuleList([SAGCNlayer(outfea,k,d) for i in range(L)])
        self.Decoder = nn.ModuleList([SAGCNlayer(outfea,k,d) for i in range(L)])
        self.STCEDA = EncoderDecoderlayer(outfea, k, d)

        self.start_emb = FeedForward([infea, outfea, outfea])
        self.end_emb = FeedForward([outfea, outfea, 1])
        self.ste_emb = STEmbedding(outfea)

        se = torch.from_numpy(SE).to(device)
        self.se = nn.Parameter(se, requires_grad=True)

    def forward(self, x, te, last=0):
        '''
        x:[B,T,N,F]
        '''
        x = x.unsqueeze(-1)

        STE = self.ste_emb(self.se, te)
        STE_P = STE[:,:12,:,:]
        STE_F = STE[:,12:,:,:]

        x = self.start_emb(x)

        for enc in self.Encoder:
            x = enc(x, STE_P)
        
        x = self.STCEDA(x, STE_P, STE_F)

        cnt=0
        for dec in self.Decoder:
            if cnt == 4:
                x = dec(x, STE_F, last)
            else:
                x = dec(x, STE_F)
            cnt+=1

        x = self.end_emb(x)

        return x.squeeze(-1)

class STEmbedding(nn.Module):
    def __init__(self, D):
        super(STEmbedding, self).__init__()
        self.ff_se = FeedForward([64,D,D])

        self.ff_te = FeedForward([295,D,D])

    def forward(self, SE, TE, T=288):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.ff_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self.ff_te(TE)
        return SE + TE

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L])

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class TCN(nn.Module):
    def __init__(self, outfea, ED):
        super(TCN, self).__init__()
        self.outfea = outfea
        self.ED = ED
        if ED:
            self.tcn = nn.ModuleList([nn.Conv2d(outfea, 2*outfea, (12,1)) for i in range(12)])
        else:
            self.tcn = nn.ModuleList([nn.Conv2d(outfea, 2*outfea, (i+1,1)) for i in range(12)])

    def forward(self, x):
        x = x.permute(0,3,1,2)
        out = []
        for i in range(12):
            if self.ED:
                out.append(self.tcn[i](x).permute(0,2,3,1))
            else:
                out.append(self.tcn[i](x[:,:,:i+1,:]).permute(0,2,3,1))
        
        out = torch.cat(out, 1)

        l, r = torch.split(out, self.outfea, -1)

        return l*torch.sigmoid(r)

class SAGCNlayer(nn.Module):
    def __init__(self, outfea, k, d):
        super(SAGCNlayer, self).__init__()
        self.kt = TCN(outfea, False)
        self.vt = TCN(outfea, False)
        self.qfc = FeedForward([2*outfea, outfea])
        self.kfc = FeedForward([2*outfea, outfea])
        self.vfc = FeedForward([2*outfea, outfea])
        self.ln = nn.LayerNorm(outfea)
        self.ff = FeedForward([outfea,outfea,outfea], True)

        self.k = k
        self.d = d

    def forward(self, x, ste, last=False):
        query = x
        key = x
        value = x

        key = self.kt(key)
        value = self.vt(value)

        query = torch.cat([query,ste],-1)
        key = torch.cat([key,ste],-1)
        value = torch.cat([value,ste],-1)

        query = self.qfc(query)
        key = self.kfc(key)
        value = self.vfc(value)

        query = torch.cat(torch.split(query, self.d, -1), 0)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,1,3,2)
        value = torch.cat(torch.split(value, self.d, -1), 0)

        attention = torch.matmul(query, key)
        attention = F.relu(attention)
        attention += torch.eye(attention.shape[-1]).to(attention.device)
        d = torch.sum(attention, -1, keepdim=True)
        attention /= d
        if last>0:
            print("!11")
            np.save(f'heatmap{last}.npy',attention.detach().cpu().numpy())
        value = torch.matmul(attention, value)

        value = torch.cat(torch.split(value, value.shape[0]//self.k, 0), -1)

        value += x

        value = self.ln(value)

        return self.ff(value)

class EncoderDecoderlayer(nn.Module):
    def __init__(self, outfea, k, d):
        super(EncoderDecoderlayer, self).__init__()
        self.vt = TCN(outfea, True)
        self.qfc = FeedForward([outfea, outfea])
        self.kfc = FeedForward([outfea, outfea])
        self.vfc = FeedForward([outfea, outfea])
        self.ln = nn.LayerNorm(outfea)
        self.ff = FeedForward([outfea,outfea,outfea], True)

        self.k = k
        self.d = d

    def forward(self, x, step, stef):
        query = stef
        key = step
        value = x

        value = self.vt(value)

        query = self.qfc(query)
        key = self.kfc(key)
        value = self.vfc(value)

        query = torch.cat(torch.split(query, self.d, -1), 0)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,1,3,2)
        value = torch.cat(torch.split(value, self.d, -1), 0)

        attention = torch.matmul(query, key)
        attention = F.relu(attention)
        attention += torch.eye(attention.shape[-1]).to(attention.device)
        d = torch.sum(attention, -1, keepdim=True)
        attention /= d
        value = torch.matmul(attention, value)

        value = torch.cat(torch.split(value, value.shape[0]//self.k, 0), -1)
        value += x

        value = self.ln(value)

        return self.ff(value)

