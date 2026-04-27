#!/usr/bin/env python3
"""
apply_feddna_native.py — 用 FedDNA 原生 Model 做解码

根因:
  - FedDNA 训练时 noise_length=155, label_length=150
  - length_adapter = Linear(155, 150) 把 encoder 输出从 155 压到 150
  - SSI-EC 的 load_pretrained_feddna 只在 sh[0]=sh[1]=max_length 时恢复，
    Linear(155, 150) 被静默跳过 → 模型架构残缺 → 解码乱码

解法:
  导入 FedDNA 原生 Model（Model.py），原样构造 + 原样加载权重。
  包一个简单适配器提供 encode_reads / decode_to_evidence 接口，
  让 step2_decode.run_feddna_decode 无需修改即可调用。

涉及文件:
  step2_runner.py — 双轨解码段的模型构造部分

用法:
  python apply_feddna_native.py
"""
import os, shutil

STEP2_FILE = "/mnt/st_data/liangxinyi/code/models/step2_runner.py"


def backup_and_replace(filepath, old, new, desc):
    with open(filepath, 'r') as f:
        content = f.read()
    if old not in content:
        if '_build_feddna_native_decoder' in content:
            print(f"  ⏭️  已打过: {desc}")
            return True
        print(f"  ❌ 未找到目标: {desc}")
        return False
    bak = filepath + '.bak_feddna_native'
    if not os.path.exists(bak):
        shutil.copy2(filepath, bak)
        print(f"  💾 备份: {bak}")
    content = content.replace(old, new, 1)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  ✅ {desc}")
    return True


def main():
    print("=" * 60)
    print("  Apply FedDNA Native Decoder")
    print("=" * 60)

    old_code = '''\
    # ── 构建 FedDNA 解码模型 ──
    _feddna_ckpt_path = getattr(args, 'feddna_checkpoint', None)
    _use_feddna_decode = False
    if _feddna_ckpt_path and os.path.exists(_feddna_ckpt_path):
        try:
            print(f"   🔧 构建 FedDNA 解码模型...")
            feddna_decode_model = Step1EvidentialModel(
                dim=model_dim, max_length=model_max_len,
                num_clusters=num_clusters, device=str(device)
            ).to(device)
            feddna_decode_model = load_pretrained_feddna(
                feddna_decode_model, _feddna_ckpt_path, device,
                max_length=model_max_len
            )
            feddna_decode_model.eval()
            _use_feddna_decode = True
            print(f"   ✅ FedDNA 解码模型就绪")
        except Exception as e:
            print(f"   ⚠️ FedDNA 解码模型失败: {e}，回退到 SSI-EC 模型")
    else:
        print(f"   ⚠️ 未提供 feddna_checkpoint，使用 SSI-EC 模型解码")'''

    new_code = '''\
    # ── 构建 FedDNA 原生解码模型（noise_length=155, label_length=150） ──
    # 根因: FedDNA 的 length_adapter 是 Linear(155, 150)，
    #        SSI-EC 的 load_pretrained_feddna 在 sh != max_length 时会跳过，
    #        导致 length_adapter 缺失、encoder↔rnnblock 连接断裂、解码乱码。
    # 解法: 直接用 FedDNA 原生 Model 构造，原样加载权重。
    #        包装一层 SSIECDecodeAdapter 提供 encode_reads / decode_to_evidence
    #        接口，step2_decode.run_feddna_decode 无需修改即可调用。
    _feddna_ckpt_path = getattr(args, 'feddna_checkpoint', None)
    _use_feddna_decode = False

    class _SSIECDecodeAdapter(nn.Module):
        """包装 FedDNA Model，提供 SSI-EC 风格的两阶段 API。"""
        def __init__(self, feddna_model, target_len):
            super().__init__()
            self.feddna_model = feddna_model
            self.target_len = target_len  # FedDNA 期望的输入长度 (noise_length)

        def encode_reads(self, reads):
            """
            reads: (B, L, 4) — SSI-EC 送进来时 L=model_max_len(=150)
            FedDNA encoder 期望输入长度 = noise_length (=155)
            需要 pad 到 155，encoder 输出还是 (B, 155, dim)
            然后走 length_adapter(155→150) 得到 (B, 150, dim)
            """
            B, L, D = reads.shape
            if L != self.target_len:
                if L < self.target_len:
                    pad = torch.zeros(B, self.target_len - L, D,
                                      device=reads.device, dtype=reads.dtype)
                    reads = torch.cat([reads, pad], dim=1)
                else:
                    reads = reads[:, :self.target_len, :]
            encoded = self.feddna_model.encoder(reads)        # (B, 155, dim)
            encoded = encoded.permute(0, 2, 1)                # (B, dim, 155)
            encoded = self.feddna_model.length_adapter(encoded)  # (B, dim, 150)
            encoded = encoded.permute(0, 2, 1)                # (B, 150, dim)
            pooled = encoded.mean(dim=1)                      # (B, dim)
            return encoded, pooled

        def decode_to_evidence(self, embeddings):
            """embeddings: (B, L, dim) → evidence (B, L, 4)"""
            evidence = self.feddna_model.rnnblock(embeddings)
            alpha = evidence + 1
            strength = alpha.sum(dim=-1)
            return evidence, strength, alpha

    def _build_feddna_native_decoder(ckpt_path, device):
        """从 FedDNA checkpoint 构造原生 Model，探测 noise/label length。"""
        from models.Model import Model as FedDNAModel, Encoder as FedDNAEncoder

        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get('model', ckpt.get('model_state_dict', ckpt))

        # 从 length_adapter 探测维度: weight shape = (out=label_length, in=noise_length)
        la_shape = sd['length_adapter.weight'].shape
        label_length = la_shape[0]
        noise_length = la_shape[1]
        print(f"      探测到 FedDNA 维度: noise_length={noise_length}, label_length={label_length}")

        # 探测 encoder dim
        enc_dim = sd['encoder.conmamba.post_norm.weight'].shape[0]

        feddna_encoder = FedDNAEncoder(dim=enc_dim)
        feddna_model = FedDNAModel(
            encoder=feddna_encoder,
            dim=enc_dim,
            noise_length=noise_length,
            label_length=label_length,
        ).to(device)
        feddna_model.load_state_dict(sd, strict=True)  # 严格加载，有问题立刻报错
        feddna_model.eval()
        return feddna_model, noise_length

    if _feddna_ckpt_path and os.path.exists(_feddna_ckpt_path):
        try:
            print(f"   🔧 构建 FedDNA 原生解码模型...")
            _feddna_native, _feddna_noise_len = _build_feddna_native_decoder(
                _feddna_ckpt_path, device
            )
            feddna_decode_model = _SSIECDecodeAdapter(
                _feddna_native, target_len=_feddna_noise_len
            ).to(device)
            feddna_decode_model.eval()
            _use_feddna_decode = True
            print(f"   ✅ FedDNA 原生模型就绪 (适配 SSI-EC 接口)")
        except Exception as e:
            import traceback
            print(f"   ⚠️ FedDNA 原生模型失败: {e}")
            traceback.print_exc()
            print(f"   ⚠️ 回退到 SSI-EC 模型解码")
    else:
        print(f"   ⚠️ 未提供 feddna_checkpoint，使用 SSI-EC 模型解码")'''

    backup_and_replace(STEP2_FILE, old_code, new_code,
                       "替换为 FedDNA 原生解码")

    print("\n" + "=" * 60)
    print("  ✅ 补丁完成")
    print("=" * 60)
    print("""
  关键改动:
    - 用 FedDNA 原生 Model（含 Linear(155,150) length_adapter）
    - 包装 _SSIECDecodeAdapter 提供 encode_reads/decode_to_evidence
    - strict=True 加载，权重有任何不匹配立刻报错

  预期:
    - 聚类保持 G老师双轨制的高分（R3 Purity=0.9699）
    - 重建 SR 恢复到 ~91% 水平（消除 decoder 退化）
""")


if __name__ == "__main__":
    main()