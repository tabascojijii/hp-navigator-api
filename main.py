"""
H!P-Navigator Backend API  v2.1.0
SQLite3-native / FTS5-JOIN-powered / Akinator engine
- pandas / numpy 依存なし
- 起動時に meta_thresholds をキャッシュ
- 全検索は view_active_originals ビュー対象
- タグ検索は tracks_fts への JOIN で実行（サブクエリ不使用）
- アキネーター次問選択はすべて SQL の GROUP BY / ABS で完結
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
# プロジェクト内の sqlite ファイル（相対パスで解決するか絶対パスで指定）
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "hp_akinator_prod.sqlite")

# ---------------------------------------------------------------------------
# グローバル状態
# ---------------------------------------------------------------------------
_con: sqlite3.Connection | None = None
THRESHOLDS: dict[str, float] = {}          # score_name → threshold_value


def _get_conn() -> sqlite3.Connection:
    """スレッドセーフなコネクションを返す (WAL モード + check_same_thread=False)"""
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
    # シャットダウン時
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
        "for Hello! Project songs (v2.1 – FTS5 JOIN edition)"
    ),
    version="2.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 共通ユーティリティ: FTS5 JOIN 節を生成
# ---------------------------------------------------------------------------
def _fts_join_clause(tag: str) -> tuple[str, str]:
    """
    FTS5 タグ検索用の JOIN 節とパラメータを返す。

    Returns
    -------
    join_sql   JOIN 節の SQL テキスト
    fts_param  MATCH に渡す文字列
    """
    join_sql = "JOIN tracks_fts f ON t.id = f.track_id"
    fts_param = f'semantic_tags:"{tag}"'
    return join_sql, fts_param


def _graduation_clause_fts() -> tuple[str, str]:
    """卒業曲特例: FTS5 JOIN で 'score_graduation > 0.7 OR タグ一致' を表現"""
    # JOIN 側は OR 条件なので、FTS MATCH でタグを引き、UNIONするより
    # score_graduation > 0.7 は WHERE 節に入れる
    # ここでは JOIN を使いつつ LEFT JOIN で both を拾う
    join_sql = "LEFT JOIN tracks_fts grad_f ON t.id = grad_f.track_id AND grad_f.semantic_tags MATCH '卒業曲'"
    where_part = "(t.score_graduation > 0.7 OR grad_f.track_id IS NOT NULL)"
    return join_sql, where_part


# ---------------------------------------------------------------------------
# Pydantic モデル
# ---------------------------------------------------------------------------
class Answer(BaseModel):
    attribute: str   # era | is_tsunku | artist_name | bpm_bucket | tags
    operator: str    # == | !=
    value: Any


class AkinatorRequest(BaseModel):
    answers: list[Answer] = []
    step: int = 1


# ---------------------------------------------------------------------------
# /search エンドポイント
# ---------------------------------------------------------------------------
@app.get("/search", summary="楽曲検索（FTS5 JOIN / スコア閾値フィルタ）")
def search(
    q:       Optional[str] = Query(None, description="キーワード（曲名・アーティスト名）"),
    tag:     Optional[str] = Query(None, description="セマンティックタグ（FTS5 MATCH）"),
    fame:    Optional[str] = Query(None, description="知名度 (standard | hidden | manic)"),
    mood:    Optional[str] = Query(None, description="感情スコアキー (euphoria, sentimental, … )"),
    bpm_min: Optional[int] = Query(None, description="最小BPM (tempo)"),
    bpm_max: Optional[int] = Query(None, description="最大BPM (tempo)"),
):
    join_clauses: list[str] = []
    where_clauses: list[str] = []
    params: list[Any] = []

    # ---- タグ検索 (FTS5 JOIN) ----
    if tag and mood != "graduation":
        join_sql, fts_param = _fts_join_clause(tag)
        join_clauses.append(join_sql)
        where_clauses.append("f.semantic_tags MATCH ?")
        params.append(fts_param)

    # ---- キーワード検索 (title / artist_name) ----
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
        grad_join, grad_where = _graduation_clause_fts()
        join_clauses.append(grad_join)
        where_clauses.append(grad_where)
    elif mood:
        score_col = f"score_{mood}"
        if score_col in THRESHOLDS:
            where_clauses.append(f"t.{score_col} >= ?")
            params.append(THRESHOLDS[score_col])

    # ---- BPM フィルタ ----
    if bpm_min is not None:
        where_clauses.append("t.tempo >= ?")
        params.append(bpm_min)
    if bpm_max is not None:
        where_clauses.append("t.tempo <= ?")
        params.append(bpm_max)

    joins = "\n        ".join(join_clauses)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT t.*
        FROM view_active_originals t
        {joins}
        {where}
        ORDER BY RANDOM()
        LIMIT 3
    """
    cur = _con.cursor()
    cur.execute(sql, params)
    return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# /akinator エンドポイント (SQL-only Akinator エンジン)
# ---------------------------------------------------------------------------
@app.post("/akinator", summary="アキネーター（SQL情報利得による次問選択）")
def run_akinator(request: AkinatorRequest):
    # ---- Step 1: answers 履歴に基づく WHERE 句の構築 ----
    join_clauses: list[str] = []
    where_clauses: list[str] = []
    params: list[Any] = []

    fts_join_idx = 0
    for ans in request.answers:
        attr = ans.attribute
        op   = ans.operator
        val  = ans.value

        if attr == "tags":
            alias = f"af{fts_join_idx}"
            fts_join_idx += 1
            fts_param = f'semantic_tags:"{val}"'
            if op == "==":
                join_clauses.append(
                    f"JOIN tracks_fts {alias} ON t.id = {alias}.track_id"
                    f" AND {alias}.semantic_tags MATCH ?"
                )
                params.append(fts_param)
            elif op == "!=":
                # NOT MATCH は FTS5 が直接非対応のため LEFT JOIN + IS NULL で表現
                join_clauses.append(
                    f"LEFT JOIN tracks_fts {alias} ON t.id = {alias}.track_id"
                    f" AND {alias}.semantic_tags MATCH ?"
                )
                params.append(fts_param)
                where_clauses.append(f"{alias}.track_id IS NULL")

        elif attr == "artist_name":
            if op == "==":
                where_clauses.append("t.artist_name = ?")
                params.append(val)
            elif op == "!=":
                where_clauses.append("t.artist_name != ?")
                params.append(val)

        elif attr == "era":
            if op == "==":
                where_clauses.append("t.era = ?")
                params.append(val)
            elif op == "!=":
                where_clauses.append("t.era != ?")
                params.append(val)

        elif attr == "bpm_bucket":
            if op == "==":
                where_clauses.append("t.bpm_bucket = ?")
                params.append(val)
            elif op == "!=":
                where_clauses.append("t.bpm_bucket != ?")
                params.append(val)

        elif attr == "is_tsunku":
            where_clauses.append("t.is_tsunku = ?")
            params.append(1 if val else 0)

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
        cur.execute(
            f"SELECT t.* FROM view_active_originals t {joins} {where} ORDER BY RANDOM() LIMIT 3",
            params,
        )
        return {
            "status": "finished",
            "remaining_count": remaining_count,
            "songs": _rows_to_dicts(cur.fetchall()),
        }

    # ---- Step 4: 次問の選定 (SQL で情報利得計算) ----
    used_attrs = {ans.attribute for ans in request.answers}

    best_diff = float("inf")
    best_question: dict | None = None

    def _check(diff: float, question: dict) -> None:
        nonlocal best_diff, best_question
        if diff < best_diff:
            best_diff = diff
            best_question = question

    # 候補 A: era
    if "era" not in used_attrs:
        cur.execute(
            f"""
            SELECT era, COUNT(*) as cnt
            FROM view_active_originals t {joins} {where}
            GROUP BY era
            ORDER BY ABS(CAST(COUNT(*) AS REAL) / {remaining_count} - 0.5)
            LIMIT 1
            """,
            params,
        )
        row = cur.fetchone()
        if row and row["cnt"] > 0:
            diff = abs(row["cnt"] / remaining_count - 0.5)
            _check(diff, {"attribute": "era", "operator": "==", "value": row["era"]})

    # 候補 B: is_tsunku
    if "is_tsunku" not in used_attrs:
        cur.execute(
            f"SELECT SUM(t.is_tsunku) as cnt_1 FROM view_active_originals t {joins} {where}",
            params,
        )
        row = cur.fetchone()
        if row and row["cnt_1"] is not None:
            diff = abs(row["cnt_1"] / remaining_count - 0.5)
            _check(diff, {"attribute": "is_tsunku", "operator": "==", "value": True})

    # 候補 C: artist_name（最も 50% 分割に近いアーティスト）
    if "artist_name" not in used_attrs:
        used_artists = [a.value for a in request.answers if a.attribute == "artist_name"]
        ex_params = list(params)
        exclude_sql = ""
        if used_artists:
            placeholders = ",".join("?" * len(used_artists))
            exclude_sql = f"AND t.artist_name NOT IN ({placeholders})"
            ex_params += used_artists
        cur.execute(
            f"""
            SELECT t.artist_name, COUNT(*) as cnt
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
            _check(diff, {"attribute": "artist_name", "operator": "==", "value": row["artist_name"]})

    # 候補 D: bpm_bucket（最も 50% 分割に近いバケット）
    if "bpm_bucket" not in used_attrs:
        cur.execute(
            f"""
            SELECT t.bpm_bucket, COUNT(*) as cnt
            FROM view_active_originals t {joins} {where}
            GROUP BY t.bpm_bucket
            ORDER BY ABS(CAST(COUNT(*) AS REAL) / {remaining_count} - 0.5)
            LIMIT 1
            """,
            params,
        )
        row = cur.fetchone()
        if row and row["cnt"] > 0:
            diff = abs(row["cnt"] / remaining_count - 0.5)
            _check(diff, {"attribute": "bpm_bucket", "operator": "==", "value": row["bpm_bucket"]})

    # 候補 E: tags（FTS5 候補タグを SQL で頻度集計 → Python 側で 50% 近似）
    # ※ FTS5 は GROUP BY 集計が困難なため、semantic_tags カラムを取得して Python 集計
    if "tags" not in used_attrs:
        used_tags = {a.value for a in request.answers if a.attribute == "tags"}
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
            for t in (x.strip() for x in raw.split(",") if x.strip()):
                if t not in used_tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1

        if tag_counts:
            best_tag = min(tag_counts, key=lambda t: abs(tag_counts[t] / remaining_count - 0.5))
            diff = abs(tag_counts[best_tag] / remaining_count - 0.5)
            _check(diff, {"attribute": "tags", "operator": "==", "value": best_tag})

    # フォールバック: 何も選べなければ最初の候補を直接返す
    if not best_question:
        cur.execute(
            f"SELECT t.title FROM view_active_originals t {joins} {where} LIMIT 1",
            params,
        )
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
# /concierge エンドポイント (ガイド付き検索)
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
    if tag and mood != "graduation":
        join_sql, fts_param = _fts_join_clause(tag)
        join_clauses.append(join_sql)
        where_clauses.append("f.semantic_tags MATCH ?")
        params.append(fts_param)

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
        grad_join, grad_where = _graduation_clause_fts()
        join_clauses.append(grad_join)
        where_clauses.append(grad_where)
    elif mood:
        score_col = f"score_{mood}"
        if score_col in THRESHOLDS:
            where_clauses.append(f"t.{score_col} >= ?")
            params.append(THRESHOLDS[score_col])

    # ---- BPM フィルタ ----
    if bpm_min is not None:
        where_clauses.append("t.tempo >= ?")
        params.append(bpm_min)
    if bpm_max is not None:
        where_clauses.append("t.tempo <= ?")
        params.append(bpm_max)

    joins = "\n        ".join(join_clauses)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    cur = _con.cursor()

    # 残件数
    cur.execute(
        f"SELECT COUNT(*) FROM view_active_originals t {joins} {where}",
        params,
    )
    remaining_count: int = cur.fetchone()[0]

    # 終了判定 (残り 20 件以下 or 5 ステップ到達)
    if remaining_count <= 20 or step >= 5:
        if remaining_count == 0:
            return {"status": "finished", "remaining_count": 0, "songs": []}
        cur.execute(
            f"SELECT t.* FROM view_active_originals t {joins} {where} ORDER BY RANDOM() LIMIT 3",
            params,
        )
        return {
            "status": "finished",
            "remaining_count": remaining_count,
            "songs": _rows_to_dicts(cur.fetchall()),
        }

    # ヒント生成: 頻出タグを集計して提示
    cur.execute(
        f"""
        SELECT t.semantic_tags
        FROM view_active_originals t {joins} {where}
        LIMIT 200
        """,
        params,
    )
    tag_counts: dict[str, int] = {}
    for row in cur.fetchall():
        raw = row["semantic_tags"] or ""
        for tg in (x.strip() for x in raw.split(",") if x.strip()):
            if not tag or tg.lower() != tag.lower():
                tag_counts[tg] = tag_counts.get(tg, 0) + 1

    if tag_counts:
        top_tags = sorted(tag_counts, key=tag_counts.__getitem__, reverse=True)[:3]
        hint = {"attribute": "tag", "options": top_tags}
    else:
        hint = {"attribute": "mood", "options": ["euphoria", "sentimental", "struggle"]}

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
    cur.execute("SELECT COUNT(*) as cnt FROM view_active_originals")
    cnt = cur.fetchone()["cnt"]
    return {
        "status": "ok",
        "db": DB_PATH,
        "active_tracks": cnt,
        "thresholds_loaded": len(THRESHOLDS),
    }
