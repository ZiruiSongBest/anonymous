#!/usr/bin/env python3
"""Metrics (PSNR/AUC/SIM) and voxel range utilities in one module.

Subcommands:
  - convert : parse range string/file to indices/xyz
  - psnr    : compute PSNR between two arrays or from MSE
  - auc     : compute pairwise AUC from pos/neg score arrays
  - sim     : compute SIM(P,G) = sum_i min(p_i, g_i)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# -------------------- Metrics --------------------

def psnr_from_mse(mse: float) -> float:
    """PSNR = -10 * log10(MSE); assumes max signal = 1."""
    mse = float(mse)
    if mse < 0:
        raise ValueError("MSE must be non-negative")
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
    mse = np.mean((pred - target) ** 2)
    return psnr_from_mse(float(mse))


def auc_pairwise(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("pos_scores and neg_scores must be non-empty")
    neg_sorted = np.sort(neg)
    counts = np.searchsorted(neg_sorted, pos, side="left")
    total = pos.size * neg.size
    return float(np.sum(counts) / total)


def sim_score(p: np.ndarray, g: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {g.shape}")
    if np.any(p < 0) or np.any(g < 0):
        raise ValueError("SIM expects non-negative inputs")
    return float(np.sum(np.minimum(p, g)))


# -------------------- Range parsing --------------------

def decode_indices_to_xyz(indices: np.ndarray, grid_size: int) -> np.ndarray:
    flat = np.asarray(indices).astype(np.int64).ravel()
    x = flat // (grid_size * grid_size)
    rem = flat % (grid_size * grid_size)
    y = rem // grid_size
    z = rem % grid_size
    return np.stack([x, y, z], axis=1)


def _iter_tokens(tokens: Iterable[str]) -> Iterable[tuple[int, int]]:
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            parts = tok.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid range token: {tok}")
            lo, hi = map(int, parts)
            if hi < lo:
                raise ValueError(f"Range must be non-decreasing: {tok}")
        else:
            lo = hi = int(tok)
        yield lo, hi


def parse_range_string_to_indices(range_str: str, grid_size: int = 32) -> np.ndarray:
    tokens = range_str.replace(",", " ").split()
    spans = list(_iter_tokens(tokens))
    counts = sum(hi - lo + 1 for lo, hi in spans)
    out = np.empty(counts, dtype=np.int64)
    pos = 0
    for lo, hi in spans:
        n = hi - lo + 1
        out[pos : pos + n] = np.arange(lo, hi + 1, dtype=np.int64)
        pos += n
    return out


def parse_range_file_to_indices(path: Path, grid_size: int = 32) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    return parse_range_string_to_indices(text, grid_size=grid_size)


def parse_range_string_to_xyz(range_str: str, grid_size: int = 32) -> np.ndarray:
    indices = parse_range_string_to_indices(range_str, grid_size=grid_size)
    return decode_indices_to_xyz(indices, grid_size=grid_size)


def parse_range_file_to_xyz(path: Path, grid_size: int = 32) -> np.ndarray:
    indices = parse_range_file_to_indices(path, grid_size=grid_size)
    return decode_indices_to_xyz(indices, grid_size=grid_size)


# -------------------- CLI helpers --------------------

def cli_convert(args: argparse.Namespace) -> int:
    if args.range_str is not None:
        indices = parse_range_string_to_indices(args.range_str, grid_size=args.grid_size)
    else:
        if not args.range_file.is_file():
            print(f"文件不存在: {args.range_file}")
            return 1
        indices = parse_range_file_to_indices(args.range_file, grid_size=args.grid_size)

    xyz = decode_indices_to_xyz(indices, grid_size=args.grid_size)

    if args.output_indices:
        np.save(args.output_indices, indices)
        print(f"已保存索引: {args.output_indices}")
    if args.output_xyz:
        np.save(args.output_xyz, xyz)
        print(f"已保存 xyz: {args.output_xyz}")

    if args.output_indices is None and args.output_xyz is None:
        print(f"解析得到 {len(indices)} 个点，网格 {args.grid_size}")
        if args.print_all or len(indices) <= 200:
            for x, y, z in xyz:
                print(f"{x} {y} {z}")
        else:
            head = 20
            for x, y, z in xyz[:head]:
                print(f"{x} {y} {z}")
            print(f"... (共 {len(indices)} 个；使用 --print-all 打印全部，或 --output-xyz 保存)")
    return 0


def cli_psnr(args: argparse.Namespace) -> int:
    if args.mse is not None:
        val = psnr_from_mse(args.mse)
    else:
        pred = np.load(args.pred)
        target = np.load(args.target)
        val = psnr(pred, target)
    print(val)
    return 0


def cli_auc(args: argparse.Namespace) -> int:
    pos = np.load(args.pos)
    neg = np.load(args.neg)
    val = auc_pairwise(pos, neg)
    print(val)
    return 0


def cli_sim(args: argparse.Namespace) -> int:
    p = np.load(args.p)
    g = np.load(args.g)
    val = sim_score(p, g)
    print(val)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metrics and range utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser("convert", help="转换范围字符串/文件为索引或 xyz")
    grp = p_conv.add_mutually_exclusive_group(required=True)
    grp.add_argument("--range-str", type=str, help="范围字符串，空格分隔，支持 a-b")
    grp.add_argument("--range-file", type=Path, help="包含范围字符串的文本文件")
    p_conv.add_argument("--grid-size", type=int, default=32, help="网格尺寸 G，默认 32")
    p_conv.add_argument("--output-xyz", type=Path, default=None, help="将 (N,3) xyz 保存为 npy")
    p_conv.add_argument("--output-indices", type=Path, default=None, help="将扁平索引保存为 npy")
    p_conv.add_argument("--print-all", action="store_true", help="打印所有坐标到标准输出")
    p_conv.set_defaults(func=cli_convert)

    p_psnr = sub.add_parser("psnr", help="计算 PSNR")
    p_psnr.add_argument("--pred", type=Path, help="预测 npy 路径")
    p_psnr.add_argument("--target", type=Path, help="真值 npy 路径")
    p_psnr.add_argument("--mse", type=float, default=None, help="直接提供 MSE 时使用")
    p_psnr.set_defaults(func=cli_psnr)

    p_auc = sub.add_parser("auc", help="计算 AUC (pairwise)")
    p_auc.add_argument("--pos", type=Path, required=True, help="正样本分数 npy")
    p_auc.add_argument("--neg", type=Path, required=True, help="负样本分数 npy")
    p_auc.set_defaults(func=cli_auc)

    p_sim = sub.add_parser("sim", help="计算 SIM")
    p_sim.add_argument("--p", type=Path, required=True, help="预测/概率 npy")
    p_sim.add_argument("--g", type=Path, required=True, help="真值/概率 npy")
    p_sim.set_defaults(func=cli_sim)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
