import sqlite3

db = sqlite3.connect(r'c:\dev\hp-navigator-api\hp_akinator_prod.sqlite')
c = db.cursor()

def test_query(fts_param):
    print(f'\nTesting FTS param: {fts_param}')
    try:
        c.execute('SELECT COUNT(*) FROM view_active_originals t JOIN tracks_fts f ON t.id = f.track_id AND f.semantic_tags MATCH ?', (fts_param,))
        print('Result:', c.fetchone()[0])
    except Exception as e:
        print('Error:', e)

test_query('semantic_tags:("16ビート" OR "クール" OR "ロック" OR "ダンス" OR "赤羽橋ファンク" OR "EDM")')
test_query('semantic_tags:"16ビート" OR semantic_tags:"クール" OR semantic_tags:"ロック" OR semantic_tags:"ダンス" OR semantic_tags:"赤羽橋ファンク" OR semantic_tags:"EDM"')
test_query('"16ビート" OR "クール" OR "ロック" OR "ダンス" OR "赤羽橋ファンク" OR "EDM"')
