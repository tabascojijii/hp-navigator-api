"""
H!P-Navigator Backend API  v2.2.0
SQLite3-native / FTS5-JOIN-powered / Tag-First Akinator engine

変更概要 (v2.1 → v2.2):
  - アキネーター: BPM質問を廃止。Era → is_tsunku → Tags → Artist の
    優先順位で情報利得を計算。Tagsは使用済みタグ単位で除外し、
    別タグへの質問は引き続き可能。
  - コンシェルジュ: 既に絞り込み済みの属性を next_hints から除外。
    タグ集計を最大500件に拡大し、指定済みタグ・mood・fame を除外した
    未使用タグ上位5件を提示。
  - 検索: BPM フィルタ適用後に 0件になる場合は BPM を緩和して再実行。
  - 全般: bpm_bucket は answers から受け取っても WHERE に適用するが、
    アキネーターの次問候補からは完全除外。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "hp_akinator_prod.sqlite")

# アキネーター: 序盤（Step≦EARLY_STEP）で era/is_tsunku を優先する閾値
EARLY_STEP_THRESHOLD = 5

# ---------------------------------------------------------------------------
# グローバル状態
# ---------------------------------------------------------------------------
_con: sqlite3.Connection | None = None
THRESHOLDS: dict[str, float] = {}   # score_name → threshold_value


def _get_conn() -> sqlite3.Connection:
    """スレッドセーフなコネクションを返す (WAL + check_same_thread=False)"""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Lifespan: 起動時に接続 & 閾値キャッシュ
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _con, THRESHOLDS
    _con = _get_conn()
    cur = _con.cursor()
    cur.execute("SELECT score_name, threshold_value FROM meta_thresholds")
    THRESHOLDS = {row["score_name"]: row["threshold_value"] for row in cur.fetchall()}
    print(f"[startup] DB connected. Thresholds loaded: {len(THRESHOLDS)} entries.")
    yield
    if _con:
        _con.close()
    print("[shutdown] DB connection closed.")


# ---------------------------------------------------------------------------
# FastAPI アプリ
# ---------------------------------------------------------------------------
app = FastAPI(
    title="H!P-Navigator Backend API",
    description=(
        "SQLite3 / FTS5 powered search & Akinator engine "
        "for Hello! Project songs (v2.2 – Tag-First edition)"
    ),
    version="2.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# ユーティリティ: SQL ビルダー
# ---------------------------------------------------------------------------

def _build_fts_join(alias: str, tag: str, left: bool = False) -> tuple[str, str]:
    """
    FTS5 JOIN 節を生成する。

    Parameters
    ----------
    alias : str   SQL テーブルエイリアス（重複しないように呼び出し元で管理）
    tag   : str   MATCH するタグ文字列
    left  : bool  True なら LEFT JOIN（NOT MATCH 用途）

    Returns
    -------
    join_sql  : JOIN 節文字列
    fts_param : MATCH に渡す文字列
    """
    join_type = "LEFT JOIN" if left else "JOIN"
    join_sql = (
        f"{join_type} tracks_fts {alias} "
        f"ON t.id = {alias}.track_id AND {alias}.semantic_tags MATCH ?"
    )

    # GPTからカンマ区切りで送られてくるケースも考慮し、カンマと全角スペースを半角スペースに置換
    clean_tag = tag.replace("　", " ").replace(",", " ")
    tags = [t.strip() for t in clean_tag.split() if t.strip()]
    
    if not tags:
        fts_param = 'semantic_tags:""'
    elif len(tags) == 1:
        fts_param = f'semantic_tags:"{tags[0]}"'
    else:
        # FTS5の古いパーサー(Vercel等の古いSQLite)でも確実に関係なく動くよう
        # semantic_tags:("A" OR "B") ではなく semantic_tags:"A" OR semantic_tags:"B" とする
        fts_param = " OR ".join(f'semantic_tags:"{t}"' for t in tags)
        
    return join_sql, fts_param


def _graduation_join_where() -> tuple[str, str]:
    """卒業曲特例: LEFT JOIN + (score > 0.7 OR タグ一致)"""
    join_sql = (
        "LEFT JOIN tracks_fts grad_f "
        "ON t.id = grad_f.track_id AND grad_f.semantic_tags MATCH '卒業曲'"
    )
    where_part = "(t.score_graduation > 0.7 OR grad_f.track_id IS NOT NULL)"
    return join_sql, where_part


def _assemble_query(
    join_clauses: list[str],
    where_clauses: list[str],
    select: str = "t.*",
    extra: str = "",
) -> str:
    """JOIN / WHERE を結合して SELECT 文を生成する。"""
    joins = "\n        ".join(join_clauses)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    return f"""
        SELECT {select}
        FROM view_active_originals t
        {joins}
        {where}
        {extra}
    """


# ---------------------------------------------------------------------------
# Pydantic モデル
# ---------------------------------------------------------------------------
class Answer(BaseModel):
    attribute: str   # era | is_tsunku | artist_name | tags
    operator: str    # == | !=
    value: Any


class AkinatorRequest(BaseModel):
    answers: list[Answer] = []
    step: int = 1


# ---------------------------------------------------------------------------
# 共通: answers → JOIN / WHERE 句の構築
# ---------------------------------------------------------------------------
def _build_from_answers(
    answers: list[Answer],
) -> tuple[list[str], list[str], list[Any]]:
    """
    アキネーターの answers リストから join_clauses / where_clauses / params を生成する。
    bpm_bucket は受け取っても WHERE に適用する（ただしアキネーターは次問候補に出さない）。
    """
    join_clauses: list[str] = []
    where_clauses: list[str] = []
    params: list[Any] = []
    fts_idx = 0

    for ans in answers:
        attr = ans.attribute
        op   = ans.operator
        val  = ans.value

        if attr == "tags":
            alias = f"af{fts_idx}"
            fts_idx += 1
            if op == "==":
                j, p = _build_fts_join(alias, val, left=False)
                join_clauses.append(j)
                params.append(p)
            elif op == "!=":
                j, p = _build_fts_join(alias, val, left=True)
                join_clauses.append(j)
                params.append(p)
                where_clauses.append(f"{alias}.track_id IS NULL")

        elif attr == "era":
            if op == "==":
                where_clauses.append("t.era = ?")
                params.append(val)
            else:
                where_clauses.append("t.era != ?")
                params.append(val)

        elif attr == "is_tsunku":
            where_clauses.append("t.is_tsunku = ?")
            params.append(1 if val else 0)

        elif attr == "artist_name":
            if op == "==":
                where_clauses.append("t.artist_name = ?")
                params.append(val)
            else:
                where_clauses.append("t.artist_name != ?")
                params.append(val)

        elif attr == "bpm_bucket":
            # 明示的に answers に含まれた場合のみ適用（次問候補には出さない）
            if op == "==":
                where_clauses.append("t.bpm_bucket = ?")
                params.append(val)
            else:
                where_clauses.append("t.bpm_bucket != ?")
                params.append(val)

    return join_clauses, where_clauses, params


# ---------------------------------------------------------------------------
# /search エンドポイント  ─ タグ優先 / BPM 軟着陸
# ---------------------------------------------------------------------------
@app.get("/search", summary="楽曲検索（FTS5 JOIN / スコア閾値フィルタ）")
def search(
    q:       Optional[str] = Query(None, description="キーワード（曲名・アーティスト名）"),
    tag:     Optional[str] = Query(None, description="セマンティックタグ（FTS5 MATCH）"),
    fame:    Optional[str] = Query(None, description="知名度 (standard | hidden | manic)"),
    mood:    Optional[str] = Query(None, description="感情スコアキー (euphoria, sentimental, … )"),
    bpm_min: Optional[int] = Query(None, description="最小BPM (tempo) ※ソート補助として利用"),
    bpm_max: Optional[int] = Query(None, description="最大BPM (tempo) ※ソート補助として利用"),
):
    cur = _con.cursor()

    def _exec_search(use_bpm: bool) -> list[dict]:
        join_clauses: list[str] = []
        where_clauses: list[str] = []
        params: list[Any] = []

        # ---- タグ検索 (FTS5 JOIN) ---- 最優先
        # JOIN 節に AND alias.semantic_tags MATCH ? を含むので WHERE 側は不要
        if tag and mood != "graduation":
            j, p = _build_fts_join("f", tag)
            join_clauses.append(j)
            params.append(p)

        # ---- キーワード検索 ----
        if q:
            where_clauses.append("(t.title LIKE ? OR t.artist_name LIKE ?)")
            wildcard = f"%{q}%"
            params += [wildcard, wildcard]

        # ---- 知名度フィルタ ----
        if fame == "standard":
            where_clauses.append("t.fame_score >= 0.3")
        elif fame == "hidden":
            where_clauses.append("t.fame_score >= 0.1 AND t.fame_score < 0.4")
        elif fame == "manic":
            where_clauses.append("t.fame_score < 0.1")

        # ---- ムードフィルタ ----
        if mood == "graduation":
            grad_j, grad_w = _graduation_join_where()
            join_clauses.append(grad_j)
            where_clauses.append(grad_w)
        elif mood:
            score_col = f"score_{mood}"
            if score_col in THRESHOLDS:
                where_clauses.append(f"t.{score_col} >= ?")
                params.append(THRESHOLDS[score_col])

        # ---- BPM フィルタ（軟着陸: 0件なら呼び出し元でリトライ）----
        if use_bpm:
            if bpm_min is not None:
                where_clauses.append("t.tempo >= ?")
                params.append(bpm_min)
            if bpm_max is not None:
                where_clauses.append("t.tempo <= ?")
                params.append(bpm_max)

        sql = _assemble_query(
            join_clauses, where_clauses,
            extra="ORDER BY RANDOM() LIMIT 3",
        )
        cur.execute(sql, params)
        return _rows_to_dicts(cur.fetchall())

    # BPM あり → 0件なら BPM を外してリトライ（軟着陸）
    results = _exec_search(use_bpm=True)
    if not results and (bpm_min is not None or bpm_max is not None):
        results = _exec_search(use_bpm=False)

    return results


# ---------------------------------------------------------------------------
# /akinator エンドポイント  ─ Tag-First 情報利得
# ---------------------------------------------------------------------------
@app.post("/akinator", summary="アキネーター（Tag-First 情報利得による次問選択）")
def run_akinator(request: AkinatorRequest):

    # ---- Step 1: answers から WHERE 構築 ----
    join_clauses, where_clauses, params = _build_from_answers(request.answers)

    joins = "\n        ".join(join_clauses)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cur = _con.cursor()

    # ---- Step 2: 残件数 ----
    cur.execute(
        f"SELECT COUNT(*) FROM view_active_originals t {joins} {where}",
        params,
    )
    remaining_count: int = cur.fetchone()[0]

    # ---- Step 3: 終了判定 ----
    if remaining_count <= 3 or request.step >= 15:
        if remaining_count == 0:
            return {"status": "finished", "remaining_count": 0, "songs": []}
        sql = _assemble_query(
            join_clauses, where_clauses,
            extra="ORDER BY RANDOM() LIMIT 3",
        )
        cur.execute(sql, params)
        return {
            "status": "finished",
            "remaining_count": remaining_count,
            "songs": _rows_to_dicts(cur.fetchall()),
        }

    # ---- Step 4: 次問選定 ─ 優先順位: era > is_tsunku > tags > artist_name ----
    # BPM (bpm_bucket) は候補から完全除外。
    # used_attrs は属性名単位で既使用を管理。
    # tags は「使用済みタグ値」単位で管理し、別タグへの質問は許可する。
    used_attrs = {ans.attribute for ans in request.answers}
    used_tags  = {ans.value for ans in request.answers if ans.attribute == "tags"}

    best_diff = float("inf")
    best_question: dict | None = None

    def _update_best(diff: float, question: dict) -> None:
        nonlocal best_diff, best_question
        if diff < best_diff:
            best_diff = diff
            best_question = question

    # ------------------------------------------------------------------
    # 候補 A: era  ─ データ充填率 100%、序盤の主力
    # ------------------------------------------------------------------
    if "era" not in used_attrs:
        cur.execute(
            f"""
            SELECT t.era, COUNT(*) AS cnt
            FROM view_active_originals t {joins} {where}
            GROUP BY t.era
            ORDER BY ABS(CAST(COUNT(*) AS REAL) / {remaining_count} - 0.5)
            LIMIT 1
            """,
            params,
        )
        row = cur.fetchone()
        if row and row["cnt"] > 0:
            diff = abs(row["cnt"] / remaining_count - 0.5)
            _update_best(diff, {"attribute": "era", "operator": "==", "value": row["era"]})

    # ------------------------------------------------------------------
    # 候補 B: is_tsunku  ─ データ充填率 100%、序盤の主力
    # ------------------------------------------------------------------
    if "is_tsunku" not in used_attrs:
        cur.execute(
            f"""
            SELECT SUM(t.is_tsunku) AS cnt_1
            FROM view_active_originals t {joins} {where}
            """,
            params,
        )
        row = cur.fetchone()
        if row and row["cnt_1"] is not None:
            diff = abs(int(row["cnt_1"]) / remaining_count - 0.5)
            _update_best(
                diff,
                {"attribute": "is_tsunku", "operator": "==", "value": True},
            )

    # ------------------------------------------------------------------
    # 候補 C: semantic_tags  ─ 中盤以降の主力
    # 使用済みタグ値（used_tags）を除いた最も 50% 分割に近いタグを探す。
    # FTS5 は GROUP BY 集計が困難なため Python 側で計算する。
    # ------------------------------------------------------------------
    cur.execute(
        f"""
        SELECT t.semantic_tags
        FROM view_active_originals t {joins} {where}
        """,
        params,
    )
    tag_counts: dict[str, int] = {}
    for row in cur.fetchall():
        raw = row["semantic_tags"] or ""
        for tg in (x.strip() for x in raw.split(",") if x.strip()):
            if tg not in used_tags:
                tag_counts[tg] = tag_counts.get(tg, 0) + 1

    if tag_counts:
        best_tag = min(
            tag_counts,
            key=lambda tg: abs(tag_counts[tg] / remaining_count - 0.5),
        )
        diff = abs(tag_counts[best_tag] / remaining_count - 0.5)
        _update_best(
            diff,
            {"attribute": "tags", "operator": "==", "value": best_tag},
        )

    # ------------------------------------------------------------------
    # 候補 D: artist_name  ─ 終盤用（候補が特定アーティストに集中したとき）
    # ------------------------------------------------------------------
    if "artist_name" not in used_attrs:
        used_artists = [a.value for a in request.answers if a.attribute == "artist_name"]
        ex_params = list(params)
        exclude_sql = ""
        if used_artists:
            ph = ",".join("?" * len(used_artists))
            exclude_sql = f"AND t.artist_name NOT IN ({ph})"
            ex_params += used_artists

        cur.execute(
            f"""
            SELECT t.artist_name, COUNT(*) AS cnt
            FROM view_active_originals t {joins}
            {where}
            {exclude_sql}
            GROUP BY t.artist_name
            ORDER BY ABS(CAST(COUNT(*) AS REAL) / {remaining_count} - 0.5)
            LIMIT 1
            """,
            ex_params,
        )
        row = cur.fetchone()
        if row and row["cnt"] > 0:
            diff = abs(row["cnt"] / remaining_count - 0.5)
            _update_best(
                diff,
                {"attribute": "artist_name", "operator": "==", "value": row["artist_name"]},
            )

    # ------------------------------------------------------------------
    # フォールバック: 候補が何もなければ最初の曲名を返す
    # ------------------------------------------------------------------
    if not best_question:
        sql = _assemble_query(join_clauses, where_clauses, select="t.title", extra="LIMIT 1")
        cur.execute(sql, params)
        row = cur.fetchone()
        best_question = {
            "attribute": "title",
            "operator": "==",
            "value": row["title"] if row else "不明",
        }

    return {
        "status": "questioning",
        "remaining_count": remaining_count,
        "next_question": best_question,
    }


# ---------------------------------------------------------------------------
# /concierge エンドポイント  ─ 重複質問防止 / 動的タグ抽出
# ---------------------------------------------------------------------------
@app.get("/concierge", summary="コンシェルジュ（段階的絞り込みガイド）")
def concierge(
    q:       Optional[str] = Query(None, description="キーワード検索"),
    tag:     Optional[str] = Query(None, description="セマンティックタグ（FTS5 MATCH）"),
    fame:    Optional[str] = Query(None, description="知名度 (standard | hidden | manic)"),
    mood:    Optional[str] = Query(None, description="感情スコアキー"),
    bpm_min: Optional[int] = Query(None, description="最小BPM"),
    bpm_max: Optional[int] = Query(None, description="最大BPM"),
    step:    int           = Query(1, description="現在の質問ステップ数"),
):
    join_clauses: list[str] = []
    where_clauses: list[str] = []
    params: list[Any] = []

    # ---- タグ検索 (FTS5 JOIN) ----
    # JOIN 節に AND alias.semantic_tags MATCH ? を含むので WHERE 側は不要
    if tag and mood != "graduation":
        j, p = _build_fts_join("f", tag)
        join_clauses.append(j)
        params.append(p)

    # ---- キーワード検索 ----
    if q:
        where_clauses.append("(t.title LIKE ? OR t.artist_name LIKE ?)")
        wildcard = f"%{q}%"
        params += [wildcard, wildcard]

    # ---- 知名度フィルタ ----
    if fame == "standard":
        where_clauses.append("t.fame_score >= 0.3")
    elif fame == "hidden":
        where_clauses.append("t.fame_score >= 0.1 AND t.fame_score < 0.4")
    elif fame == "manic":
        where_clauses.append("t.fame_score < 0.1")

    # ---- ムードフィルタ ----
    if mood == "graduation":
        grad_j, grad_w = _graduation_join_where()
        join_clauses.append(grad_j)
        where_clauses.append(grad_w)
    elif mood:
        score_col = f"score_{mood}"
        if score_col in THRESHOLDS:
            where_clauses.append(f"t.{score_col} >= ?")
            params.append(THRESHOLDS[score_col])

    # ---- BPM フィルタ（軟着陸付き）----
    bpm_requested = bpm_min is not None or bpm_max is not None

    def _base_query_with_bpm(use_bpm: bool) -> tuple[list[str], list[str], list[Any]]:
        """BPM の有無だけ違うクローンを作る"""
        wc = list(where_clauses)
        pr = list(params)
        if use_bpm:
            if bpm_min is not None:
                wc.append("t.tempo >= ?")
                pr.append(bpm_min)
            if bpm_max is not None:
                wc.append("t.tempo <= ?")
                pr.append(bpm_max)
        return list(join_clauses), wc, pr

    cur = _con.cursor()

    # BPM 込みで件数確認
    jc_bpm, wc_bpm, pr_bpm = _base_query_with_bpm(use_bpm=bpm_requested)
    joins_bpm = "\n        ".join(jc_bpm)
    where_bpm = ("WHERE " + " AND ".join(wc_bpm)) if wc_bpm else ""

    cur.execute(
        f"SELECT COUNT(*) FROM view_active_originals t {joins_bpm} {where_bpm}",
        pr_bpm,
    )
    remaining_count: int = cur.fetchone()[0]

    # 軟着陸: BPM あり → 0件 → BPM なしで再カウント
    effective_jc, effective_wc, effective_params = jc_bpm, wc_bpm, pr_bpm
    if remaining_count == 0 and bpm_requested:
        jc_nb, wc_nb, pr_nb = _base_query_with_bpm(use_bpm=False)
        joins_nb = "\n        ".join(jc_nb)
        where_nb = ("WHERE " + " AND ".join(wc_nb)) if wc_nb else ""
        cur.execute(
            f"SELECT COUNT(*) FROM view_active_originals t {joins_nb} {where_nb}",
            pr_nb,
        )
        remaining_count = cur.fetchone()[0]
        effective_jc, effective_wc, effective_params = jc_nb, wc_nb, pr_nb

    effective_joins = "\n        ".join(effective_jc)
    effective_where = ("WHERE " + " AND ".join(effective_wc)) if effective_wc else ""

    # ---- 終了判定 (残り 20 件以下 or 5 ステップ到達) ----
    if remaining_count <= 20 or step >= 5:
        if remaining_count == 0:
            return {"status": "finished", "remaining_count": 0, "songs": []}
        sql = _assemble_query(
            effective_jc, effective_wc,
            extra="ORDER BY RANDOM() LIMIT 3",
        )
        cur.execute(sql, effective_params)
        return {
            "status": "finished",
            "remaining_count": remaining_count,
            "songs": _rows_to_dicts(cur.fetchall()),
        }

    # ---- ヒント生成: 既使用属性を除外した上位タグ動的抽出 ----
    # 既に絞り込みに使われている属性を「使用済み」として記録
    already_used_tags: set[str] = set()
    if tag:
        already_used_tags.add(tag.lower())

    # mood や fame が指定されている場合、それ系のヒントは出さない
    # → semantic_tags の中から「まだ指定していないタグ」の上位5件を提示

    cur.execute(
        f"""
        SELECT t.semantic_tags
        FROM view_active_originals t {effective_joins} {effective_where}
        LIMIT 500
        """,
        effective_params,
    )
    tag_freq: dict[str, int] = {}
    for row in cur.fetchall():
        raw = row["semantic_tags"] or ""
        for tg in (x.strip() for x in raw.split(",") if x.strip()):
            # 既に絞り込みで使用済みのタグ（大文字小文字無視）を除外
            if tg.lower() not in already_used_tags:
                tag_freq[tg] = tag_freq.get(tg, 0) + 1

    if tag_freq:
        top_tags = sorted(tag_freq, key=tag_freq.__getitem__, reverse=True)[:5]
        hint = {"attribute": "tag", "options": top_tags}
    else:
        # タグが尽きた場合は mood/fame を提案（ただし既に使用済みなら除く）
        fallback_moods = [
            m for m in ["euphoria", "sentimental", "struggle", "energy_burst"]
            if m != mood
        ]
        hint = {"attribute": "mood", "options": fallback_moods[:3]}

    return {
        "status": "questioning",
        "remaining_count": remaining_count,
        "next_hints": hint,
    }


# ---------------------------------------------------------------------------
# /health エンドポイント
# ---------------------------------------------------------------------------
@app.get("/health", summary="ヘルスチェック", include_in_schema=False)
def health():
    cur = _con.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM view_active_originals")
    cnt = cur.fetchone()["cnt"]
    return {
        "status": "ok",
        "db": DB_PATH,
        "active_tracks": cnt,
        "thresholds_loaded": len(THRESHOLDS),
    }
