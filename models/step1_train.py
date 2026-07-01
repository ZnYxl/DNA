# models/step1_train.py
"""
Step1 training entry point.

Per round: build/load consensus targets, create the dataset with cluster-level
dynamic sampling, hot-start from FedDNA (R1) or the previous round (R2+), train the
evidential model (uncertainty-weighted contrastive + masked Bayes-risk + annealed
KL), run a short rnnblock calibration phase, and save the checkpoint.
"""
import torch
import torch.optim as optim
import os
import sys
import time
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.step1_model import Step1EvidentialModel, load_pretrained_feddna, masked_bayes_risk
from models.step1_data  import CloverDataLoader, Step1Dataset, create_dynamic_sampler, seq_to_onehot
from models.step1_visualizer import Step1Visualizer


# ---------------------------------------------------------------------------
# Hot-start hyperparameter defaults (overridable via args from main_loop.py)
# ---------------------------------------------------------------------------
_DEFAULT_ROUND1_EPOCHS = 10
_DEFAULT_ROUND1_LR     = 1e-4
_DEFAULT_ROUND2_EPOCHS = 10
_DEFAULT_ROUND2_LR     = 5e-5   # Decoder lr; Encoder uses 5e-6 (differential)


class ListBatchSampler:
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


def train_step1(args):
    """Step 1 training main function."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    round_idx    = getattr(args, 'round_idx', 1)
    prev_state   = getattr(args, 'prev_state', None)
    training_cap = getattr(args, 'training_cap', 2000000)

    # =====================================================================
    # 1. Load data
    # =====================================================================
    print("\n" + "=" * 60)
    print("Data loading")
    print("=" * 60)

    labels_path = getattr(args, 'refined_labels', None)
    data_loader = CloverDataLoader(args.experiment_dir, labels_path=labels_path)

    # =====================================================================
    # 2. Compute or load consensus_dict
    # =====================================================================
    print("\n" + "=" * 60)
    print("Consensus computation")
    print("=" * 60)

    consensus_path = getattr(args, 'consensus_path', None)
    consensus_dict: Dict[int, torch.Tensor] = {}

    if consensus_path and os.path.exists(consensus_path):
        # Round 2+: load from Step 2 output
        print(f"   loading consensus from: {consensus_path}")
        consensus_dict = torch.load(consensus_path, map_location='cpu')
        consensus_dict = {int(k): v for k, v in consensus_dict.items()}  # force int keys
        print(f"   loaded consensus for {len(consensus_dict)} clusters")
    else:
        # Round 1: build from ref.txt (preprocessing already did Clover majority vote).
        # data_loader.ref_seqs = {cluster_id: ref_seq_str}, loaded in _load_all_data.
        print(f"   Round {round_idx}: building consensus_dict from ref.txt ...")
        if not data_loader.ref_seqs:
            raise RuntimeError(
                "ref.txt not loaded. Check that CloverDataLoader._load_all_data reads ref.txt correctly."
            )
        for cid, seq in data_loader.ref_seqs.items():
            consensus_dict[cid] = seq_to_onehot(seq, args.max_length)  # (L, 4)
        print(f"   consensus_dict: {len(consensus_dict)} clusters")

    # =====================================================================
    # 3. Read cluster_change_info (Round 2+ cluster-level sampling basis)
    # =====================================================================
    cluster_change_info = getattr(args, 'cluster_change_info', None)
    if cluster_change_info is not None:
        hard_count = sum(1 for v in cluster_change_info.values() if v >= 0.05)
        easy_count = len(cluster_change_info) - hard_count
        print(f"   cluster_change_info: hard={hard_count}, easy={easy_count}")

    # =====================================================================
    # 4. Create dataset (carries consensus_dict + cluster_change_info)
    # =====================================================================
    print("\n" + "=" * 60)
    print("Dataset creation")
    print("=" * 60)

    dataset = Step1Dataset(
        data_loader,
        max_len=args.max_length,
        training_cap=training_cap,
        inference_mode=False,
        round_idx=round_idx,
        consensus_dict=consensus_dict,
        cluster_change_info=cluster_change_info,
        cv_threshold=getattr(args, 'cv_threshold', 0.3),
        max_reads_per_cluster=getattr(args, 'max_reads_per_cluster', 30),  # aligned with FedDNA
    )

    # =====================================================================
    # 5. Create model
    # =====================================================================
    print("\n" + "=" * 60)
    print("Model creation")
    print("=" * 60)

    num_clover_clusters = len(set(l for l in data_loader.clover_labels if l >= 0))
    num_clusters    = max(num_clover_clusters, args.min_clusters)

    model = Step1EvidentialModel(
        dim=args.dim,
        max_length=args.max_length,
        num_clusters=num_clusters,
        device=str(device),
        cl_mode=getattr(args, 'cl_mode', 'ours'),   # ablation flag
    ).to(device)

    print(f"   model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   current clusters: {num_clover_clusters}")
    print(f"   contrastive mode: {getattr(args, 'cl_mode', 'ours')}")

    # =====================================================================
    # 6. Hot-start weight loading
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"Hot start (Round {round_idx})")
    print("=" * 60)

    if round_idx <= 1:
        feddna_ckpt = getattr(args, 'feddna_checkpoint', None)
        if feddna_ckpt and os.path.exists(feddna_ckpt):
            model = load_pretrained_feddna(model, feddna_ckpt, device,
                                           max_length=args.max_length)
        else:
            print(f"   [warn] no pretrained weights found, using random init")
    else:
        prev_ckpt = getattr(args, 'prev_checkpoint', None)
        if prev_ckpt and os.path.exists(prev_ckpt):
            try:
                import torch.nn as nn_
                ckpt = torch.load(prev_ckpt, map_location=device)
                sd   = ckpt.get('model_state_dict', ckpt)

                if 'length_adapter.weight' in sd:
                    sh = sd['length_adapter.weight'].shape
                    # Both input and output dims must match. If only the output dim
                    # were checked, a checkpoint with noise_length != label_length
                    # (e.g. 155->150) would build nn.Linear(155, 150) while the
                    # encoder output length is 150, causing a matmul dim mismatch.
                    if sh[1] == args.max_length and sh[0] == args.max_length:
                        model.length_adapter = nn_.Linear(sh[1], sh[0]).to(device)
                        print(f"   pre-initialized length_adapter: Linear({sh[1]}, {sh[0]})")
                    elif sh[1] != args.max_length or sh[0] != args.max_length:
                        print(f"   [warn] checkpoint length_adapter dim {sh} "
                              f"incompatible with max_length={args.max_length}, skipped")

                model.load_state_dict(sd, strict=False)
                print(f"   loaded previous-round weights")

                # Round 2+: Encoder is not frozen; differential learning rates are
                # used instead (Encoder 5e-6, Decoder 5e-5). Freezing the encoder
                # made R2/R3 fail to converge.
                total_params = sum(p.numel() for p in model.parameters())
                enc_params = sum(p.numel() for p in model.encoder.parameters())
                print(f"   Encoder unfrozen (differential learning rate)")
                print(f"   params: total={total_params:,}, Encoder={enc_params:,}, Decoder={total_params - enc_params:,}")
            except Exception as e:
                print(f"   [warn] load failed: {e}, using random init")
        else:
            print(f"   [warn] no previous checkpoint, random init")

    # =====================================================================
    # 7. Optimizer + scheduler
    # =====================================================================
    epochs = getattr(args, 'round1_epochs', _DEFAULT_ROUND1_EPOCHS) if round_idx <= 1 else getattr(args, 'round2_epochs', _DEFAULT_ROUND2_EPOCHS)
    lr     = getattr(args, 'round1_lr',     _DEFAULT_ROUND1_LR)     if round_idx <= 1 else getattr(args, 'round2_lr',     _DEFAULT_ROUND2_LR)

    print(f"\n   training hyperparams: epochs={epochs}, lr={lr}")

    # Differential learning rate: Encoder fine-tunes at a tiny lr, Decoder uses a
    # normal lr to adapt to the new consensus.
    if round_idx <= 1:
        # Round 1: single lr for all parameters
        trainable_params = list(model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    else:
        # Round 2+: Encoder 5e-6, Decoder 5e-5
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
        enc_lr = 5e-6
        dec_lr = lr  # = _DEFAULT_ROUND2_LR = 5e-5
        trainable_params = encoder_params + decoder_params  # for clip_grad_norm
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': enc_lr},
            {'params': decoder_params, 'lr': dec_lr},
        ], weight_decay=args.weight_decay)
        print(f"   differential lr: Encoder={enc_lr}, Decoder={dec_lr}")
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # =====================================================================
    # 8. Training loop
    # =====================================================================
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    model.train()
    # Encoder unfrozen: all submodules in train mode (BN updates stats, Dropout on),
    # which is the correct configuration for the differential-lr scheme.

    training_history = {
        'total_loss': [], 'avg_strength': [], 'high_conf_ratio': [],
        'contrastive_loss': [], 'reconstruction_loss': [], 'kl_loss': [],
        'u_epi_mean': [], 'u_ale_mean': [], 'queue_count': []
    }

    for epoch in range(epochs):
        start_time = time.time()

        batch_indices_list = create_dynamic_sampler(
            dataset,
            batch_size=args.batch_size,
            max_clusters_per_batch=args.max_clusters_per_batch,
            state_path=prev_state,
            round_idx=round_idx
        )

        batch_sampler = ListBatchSampler(batch_indices_list)
        train_loader  = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True
        )

        epoch_loss  = epoch_con = epoch_rec = epoch_kl = 0
        epoch_str   = epoch_hc = epoch_u_epi = epoch_u_ale = epoch_qc = 0
        epoch_w_cc = epoch_w_da = epoch_cos_pos = epoch_cos_neg = 0
        epoch_w_cc_cnt = epoch_w_da_cnt = epoch_cos_pos_cnt = epoch_cos_neg_cnt = 0
        num_batches = 0
        total_batches = len(batch_indices_list)

        print(f"\nEpoch {epoch + 1}/{epochs} ({total_batches} batches)")

        for i, batch_data in enumerate(train_loader):
            reads_batch      = batch_data['encoding'].to(device)
            labels_batch     = batch_data['clover_label'].to(device)
            consensus_batch  = batch_data['consensus_target'].to(device)

            loss_dict, outputs = model(reads_batch, labels_batch, consensus_batch, epoch=epoch, round_idx=round_idx)

            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss  += loss_dict['total'].item()
            epoch_con   += loss_dict['contrastive'].item()
            epoch_rec   += loss_dict['reconstruction'].item()
            epoch_kl    += loss_dict['kl_divergence'].item()
            epoch_str   += outputs['avg_strength']
            epoch_hc    += outputs['high_conf_ratio']
            epoch_u_epi += outputs.get('u_epi_mean', 0.0)
            epoch_u_ale += outputs.get('u_ale_mean', 0.0)
            epoch_qc    += outputs.get('queue_count', 0)

            # Four probes accumulated independently to avoid nan cross-contamination
            wcc = outputs.get('w_clean_clean', float('nan'))
            wda = outputs.get('w_dirty_any',   float('nan'))
            cp  = outputs.get('cos_sim_pos',   float('nan'))
            cn  = outputs.get('cos_sim_neg',   float('nan'))
            if wcc == wcc: epoch_w_cc += wcc; epoch_w_cc_cnt += 1
            if wda == wda: epoch_w_da += wda; epoch_w_da_cnt += 1
            if cp == cp:   epoch_cos_pos += cp; epoch_cos_pos_cnt += 1
            if cn == cn:   epoch_cos_neg += cn; epoch_cos_neg_cnt += 1

            num_batches += 1

            if (i + 1) % 50 == 0:
                print(f"   [Batch {i+1}/{total_batches}] "
                      f"Loss: {loss_dict['total'].item():.4f} | "
                      f"Str: {outputs['avg_strength']:.1f} | "
                      f"U_epi: {outputs.get('u_epi_mean',0):.4f}",
                      end='\r')

        scheduler.step()
        epoch_time = time.time() - start_time
        avg = lambda x: x / max(num_batches, 1)

        training_history['total_loss'].append(avg(epoch_loss))
        training_history['contrastive_loss'].append(avg(epoch_con))
        training_history['reconstruction_loss'].append(avg(epoch_rec))
        training_history['kl_loss'].append(avg(epoch_kl))
        training_history['avg_strength'].append(avg(epoch_str))
        training_history['high_conf_ratio'].append(avg(epoch_hc))
        training_history['u_epi_mean'].append(avg(epoch_u_epi))
        training_history['u_ale_mean'].append(avg(epoch_u_ale))
        training_history['queue_count'].append(avg(epoch_qc))

        print(f"\n   Epoch {epoch+1} ({epoch_time:.1f}s) | "
              f"Loss: {avg(epoch_loss):.4f} | Str: {avg(epoch_str):.1f} | "
              f"Recon: {avg(epoch_rec):.4f} | U_epi: {avg(epoch_u_epi):.4f}")

        # Four probes reported independently
        avg_wcc = epoch_w_cc / max(epoch_w_cc_cnt, 1)
        avg_wda = epoch_w_da / max(epoch_w_da_cnt, 1)
        avg_cp  = epoch_cos_pos / max(epoch_cos_pos_cnt, 1)
        avg_cn  = epoch_cos_neg / max(epoch_cos_neg_cnt, 1)
        wcc_s = f"{avg_wcc:.4f}" if epoch_w_cc_cnt > 0 else "nan"
        wda_s = f"{avg_wda:.4f}" if epoch_w_da_cnt > 0 else "nan"
        cp_s  = f"{avg_cp:.4f}"  if epoch_cos_pos_cnt > 0 else "nan"
        cn_s  = f"{avg_cn:.4f}"  if epoch_cos_neg_cnt > 0 else "nan"
        if epoch_w_cc_cnt > 0 or epoch_w_da_cnt > 0:
            ratio_s = f"{avg_wcc / max(avg_wda, 1e-6):.1f}x" if epoch_w_da_cnt > 0 else "nan"
            print(f"   probe A | w(clean-clean): {wcc_s}  "
                  f"w(dirty-any): {wda_s}  ratio: {ratio_s}")
        if epoch_cos_pos_cnt > 0 or epoch_cos_neg_cnt > 0:
            margin_s = f"{avg_cp - avg_cn:.4f}" if (epoch_cos_pos_cnt > 0 and epoch_cos_neg_cnt > 0) else "nan"
            print(f"   probe B | cos_pos: {cp_s}  cos_neg: {cn_s}  margin: {margin_s}")

    # =====================================================================
    # 8.5 Calibration phase: rnnblock calibration
    # =====================================================================
    # Contrastive training reorganizes the encoder feature space (cos_neg 0.98->0.04);
    # the rnnblock cannot fully adapt to the final feature space in time. Freeze the
    # encoder and fine-tune only the rnnblock with recon_loss so it learns to decode
    # in a static space -- analogous to linear-probe fine-tuning after SimCLR/MoCo.
    calib_epochs = getattr(args, 'calib_epochs', 3)
    calib_lr     = getattr(args, 'calib_lr', 2e-5)

    if calib_epochs > 0:
        print("\n" + "=" * 60)
        print(f"Calibration phase: rnnblock ({calib_epochs} epochs, lr={calib_lr})")
        print("=" * 60)

        # Freeze Encoder + Length Adapter
        for p in model.encoder.parameters():
            p.requires_grad = False
        if model.length_adapter is not None:
            for p in model.length_adapter.parameters():
                p.requires_grad = False

        # Ensure rnnblock is trainable
        calib_params = list(model.rnnblock.parameters())
        for p in calib_params:
            p.requires_grad = True

        trainable_count = sum(p.numel() for p in calib_params if p.requires_grad)
        frozen_count    = sum(p.numel() for p in model.parameters()) - trainable_count
        print(f"   frozen: {frozen_count:,} params (Encoder + Length Adapter)")
        print(f"   trainable: {trainable_count:,} params (RNNBlock)")

        # Fresh optimizer (discard momentum contamination from the main training phase)
        calib_optimizer = optim.AdamW(calib_params, lr=calib_lr, weight_decay=1e-4)

        model.train()
        model.encoder.eval()  # BN stats fixed, Dropout off

        for calib_epoch in range(calib_epochs):
            calib_start = time.time()

            # Reuse the main training dynamic sampler
            calib_batches = create_dynamic_sampler(
                dataset,
                batch_size=args.batch_size,
                max_clusters_per_batch=args.max_clusters_per_batch,
                state_path=prev_state,
                round_idx=round_idx
            )
            calib_sampler = ListBatchSampler(calib_batches)
            calib_loader  = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=calib_sampler,
                num_workers=4,
                pin_memory=True
            )

            calib_loss_sum = 0
            calib_str_sum  = 0
            calib_n        = 0

            for i, batch_data in enumerate(calib_loader):
                reads_batch     = batch_data['encoding'].to(device)
                consensus_batch = batch_data['consensus_target'].to(device)

                # encoder -> decoder -> recon_loss only
                embeddings, _ = model.encode_reads(reads_batch)
                evidence, strength, alpha = model.decode_to_evidence(embeddings)

                recon_loss = masked_bayes_risk(evidence, consensus_batch)

                calib_optimizer.zero_grad()
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(calib_params, max_norm=1.0)
                calib_optimizer.step()

                calib_loss_sum += recon_loss.item()
                calib_str_sum  += strength.mean().item()
                calib_n        += 1

                if (i + 1) % 100 == 0:
                    print(f"   [Calib Batch {i+1}/{len(calib_batches)}] "
                          f"Recon: {recon_loss.item():.4f} | "
                          f"Str: {strength.mean().item():.1f}", end='\r')

            calib_time = time.time() - calib_start
            avg_loss = calib_loss_sum / max(calib_n, 1)
            avg_str  = calib_str_sum / max(calib_n, 1)
            print(f"\n   Calib Epoch {calib_epoch+1}/{calib_epochs} ({calib_time:.1f}s) | "
                  f"Recon: {avg_loss:.4f} | Str: {avg_str:.1f}")

        # Restore all parameters to trainable (does not affect the subsequent save)
        for p in model.parameters():
            p.requires_grad = True

        print(f"   calibration done")

    # =====================================================================
    # 9. Save checkpoint
    # =====================================================================
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "step1_final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'training_history': training_history,
        'round_idx': round_idx,
    }, checkpoint_path)
    print(f"\n   checkpoint saved: {checkpoint_path}")

    # Visualization
    try:
        viz = Step1Visualizer(output_dir)
        viz.plot_training_losses(training_history)
        viz.plot_evidence_stats(training_history)
        viz.save_config(args)
    except Exception as e:
        print(f"   [warn] visualization skipped: {e}")

    return checkpoint_path