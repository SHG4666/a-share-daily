"""
全市场历史行情缓存

逐只拉取历史数据很慢（全市场约 5000 次请求，约 40 分钟），所以：
  1. 首次全量拉取，保存到 cache/hist.pkl.gz（GitHub Actions cache 跨次运行保留，不进 git）
  2. 之后每天用全市场快照（约 20 次请求拿到所有股票）给每只股票追加当天 K 线
  3. 如果快照里的“昨收”和缓存里的最后收盘价对不上（除权除息、复权口径变化、数据缺口），
     说明这只股票的前复权历史已经变了，标记为过期，重新完整拉取
  4. 拉取有时间预算，超时就先保存已拉到的，下次运行继续补齐

成交量统一用“手”。
"""
import datetime as dt
import pickle
import gzip
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import data

PATH = Path(__file__).parent / "cache" / "hist.pkl.gz"
KEEP_ROWS = 600
COLS = ["date", "close", "high", "low", "vol"]


class Store:
    def __init__(self):
        self.h: dict[str, pd.DataFrame] = {}
        self.stale: set[str] = set()
        self.loaded = 0

    # ------------------------------------------------------------ 持久化
    def load(self):
        if PATH.exists():
            try:
                with gzip.open(PATH, "rb") as f:
                    loaded = pickle.load(f)
                self.h.clear()          # 原地更新：main._HIST 引用的是同一个字典
                self.h.update(loaded)
                self.loaded = len(self.h)
                print(f"历史缓存：载入 {len(self.h)} 只")
                self.repair_units()
            except Exception as e:  # noqa: BLE001
                print("历史缓存损坏，重建:", e)
                self.h.clear()
        else:
            print("历史缓存：不存在，将从头构建")
        return self

    def repair_units(self):
        """修复历史上用错单位追加的成交量：腾讯快照对科创板报“股”，曾被当成“手”追加，导致当天量放大 100 倍。
        判定：科创板股票某天成交量 > 前 20 天中位数的 30 倍，且除以 100 后回到正常范围 → 除以 100"""
        fixed = 0
        for code, h in self.h.items():
            if not code.startswith(("688", "689")) or h is None or len(h) < 22:
                continue
            v = h["vol"].to_numpy(float)
            med = pd.Series(v).shift(1).rolling(20).median().to_numpy()
            idx = np.where((v > 30 * med) & (v / 100 < 30 * med))[0]
            idx = idx[idx >= len(v) - 30]  # 只看最近 30 天（增量追加的部分）
            if len(idx):
                v = v.copy(); v[idx] = v[idx] / 100
                h["vol"] = v
                fixed += len(idx)
        if fixed:
            print(f"历史缓存：修复科创板成交量单位 {fixed} 条")

    def save(self):
        PATH.parent.mkdir(exist_ok=True)
        tmp = PATH.with_suffix(".tmp")
        with gzip.open(tmp, "wb", compresslevel=5) as f:
            pickle.dump(self.h, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(PATH)
        print(f"历史缓存：保存 {len(self.h)} 只，{PATH.stat().st_size / 1e6:.0f} MB")

    # ------------------------------------------------------------ 每日增量
    def append_snapshot(self, spot: pd.DataFrame, day: dt.date):
        """用当天快照追加 K 线；昨收对不上的标记为过期"""
        need = {"昨收", "最高", "最低", "成交量"}
        if not need.issubset(spot.columns):
            print("快照缺少 OHLC 字段，跳过增量追加")
            return
        appended = 0
        for r in spot.itertuples(index=False):
            code = r.代码
            h = self.h.get(code)
            if h is None:
                continue
            last = h["date"].iloc[-1]
            if last >= day:
                continue
            c, pc, hi, lo, v = r.最新价, r.昨收, r.最高, r.最低, r.成交量
            if not all(np.isfinite([c, pc, hi, lo, v])) or c <= 0 or v <= 0:
                continue  # 停牌或数据缺失
            if not (lo <= c <= hi):
                self.stale.add(code); continue
            if abs(pc / h["close"].iloc[-1] - 1) > 0.003:
                self.stale.add(code); continue  # 除权除息 / 中间缺了交易日
            row = pd.DataFrame([[day, c, hi, lo, v]], columns=COLS)
            self.h[code] = pd.concat([h, row], ignore_index=True).tail(KEEP_ROWS).reset_index(drop=True)
            appended += 1
        print(f"历史缓存：追加当天 K 线 {appended} 只，需重拉 {len(self.stale)} 只")

    # ------------------------------------------------------------ 补齐
    def ensure(self, codes, day: dt.date, budget_sec: float, traded: set | None = None, workers: int = 6):
        """确保 codes 都有截至 day 的历史；缺失、过期、或当天有交易但缓存没跟上的，在时间预算内补拉。
        traded：当天有成交的股票（停牌股缓存停在停牌前是正常的，不必重拉）"""
        def outdated(c):
            return self.h[c]["date"].iloc[-1] < day and (traded is None or c in traded)
        todo = [c for c in dict.fromkeys(codes) if c not in self.h or c in self.stale or outdated(c)]
        if not todo:
            return 0
        t0, done, fail = time.time(), 0, 0

        def one(code):
            if time.time() - t0 > budget_sec:
                return code, "skip"
            time.sleep(0.15)
            try:
                return code, data.get_hist(code)
            except Exception as e:  # noqa: BLE001
                return code, e

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, c) for c in todo]
            for f in as_completed(futs):
                code, h = f.result()
                if isinstance(h, pd.DataFrame) and len(h):
                    self.h[code] = h[COLS].tail(KEEP_ROWS).reset_index(drop=True)
                    self.stale.discard(code)
                    done += 1
                    if done % 500 == 0:
                        print(f"  已拉取 {done}/{len(todo)}，用时 {time.time() - t0:.0f}s")
                elif not (isinstance(h, str) and h == "skip"):
                    fail += 1
        left = len(todo) - done - fail
        print(f"历史数据：需补 {len(todo)} 只，成功 {done}，失败 {fail}，超出时间预算未拉 {left}，"
              f"用时 {time.time() - t0:.0f}s")
        return left

    def get(self, code):
        return self.h.get(code)
