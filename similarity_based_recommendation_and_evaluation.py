# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import timedelta
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. 모델 로드
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 2. 데이터 로드 및 전처리
# =========================
article = pd.read_csv("/content/drive/MyDrive/BKBS/news_bind.csv")
article = article.drop_duplicates(subset=["title", "source"])
article['published'] = pd.to_datetime(article['published'], format="%Y-%m-%d")


# =========================
# 3. 추천 함수
# =========================
def recommend_articles_by_cosine(base_article, articles_df, model, top_n=5, time_window=30):
    df = articles_df.copy()
    base = base_article.copy()

    # 후보 기사 필터링
    start_date = base['published'] - timedelta(days=time_window)
    end_date = base['published'] + timedelta(days=time_window)

    candidates = df[(df['published'] >= start_date) & (df['published'] <= end_date)].reset_index(drop=True)

    # 기준 기사 제외
    if 'context_id' in base and 'context_id' in candidates.columns:
        candidates = candidates[candidates['context_id'] != base['context_id']]
    else:
        candidates = candidates[candidates['title'] != base['title']]

    if candidates.empty:
        return pd.DataFrame(columns=['recommend_id', 'title', 'context', 'published', 'source', 'similarity', 'base_id'])

    # 텍스트 임베딩
    base_text = base['title'] + " " + base['context']
    base_emb = model.encode(base_text, convert_to_tensor=True)

    candidate_texts = candidates['title'] + " " + candidates['context']
    candidate_embs = model.encode(candidate_texts.tolist(), convert_to_tensor=True)

    # 코사인 유사도
    cos_scores = util.pytorch_cos_sim(base_emb, candidate_embs)[0]
    candidates['similarity'] = cos_scores.cpu().numpy()

    # 상위 N개
    top_articles = candidates.sort_values(by='similarity', ascending=False).head(top_n)
    top_articles['base_id'] = base['context_id']
    top_articles['recommend_id'] = top_articles['context_id']

    return top_articles[['recommend_id', 'title', 'context', 'published', 'source', 'similarity', 'base_id']]


# =========================
# 4. 평가 지표 함수
# =========================
def diversity_score(recommend_articles, model):
    texts = (recommend_articles['title'] + " " + recommend_articles['context']).tolist()
    if len(texts) < 2:
        return 0.0
    embeddings = model.encode(texts, convert_to_tensor=False)
    sim_matrix = cosine_similarity(embeddings)
    n = len(sim_matrix)
    upper = sim_matrix[np.triu_indices(n, k=1)]
    avg_sim = np.mean(upper)
    return (1 - avg_sim).round(4)


def cgi_score(recommend_articles):
    publisher_list = recommend_articles['source'].tolist()
    if len(publisher_list) < 2:
        return 0.0
    counts = np.array(list({p: publisher_list.count(p) for p in set(publisher_list)}.values()))
    counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(counts)
    S_n = cum_counts[-1]
    numerator = 2 * np.sum(cum_counts[:-1])
    denominator = n * S_n - S_n
    gini = numerator / denominator if denominator != 0 else 0.0
    return (1 - gini).round(4)


def per_seed_scores(df, model, seed_col='seed_article_id'):
    rows = []
    for seed, group in df.groupby(seed_col):
        texts_count = len(group)
        if texts_count < 2:
            rows.append((seed, np.nan, np.nan, texts_count))
            continue
        d = diversity_score(group, model)
        c = cgi_score(group)
        rows.append((seed, d, c, texts_count))
    return pd.DataFrame(rows, columns=[seed_col, 'diversity', 'cgi', 'n_recommendations'])


def bootstrap_mean_ci(values, n_boot=1000, ci=0.95, random_state=None):
    vals = np.array(values)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return {'mean': np.nan, 'std': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n': 0}
    rng = np.random.RandomState(random_state)
    boot_means = [np.mean(rng.choice(vals, size=n, replace=True)) for _ in range(n_boot)]
    mean = np.mean(vals).round(4)
    std = np.std(vals, ddof=1).round(4)
    alpha = 1 - ci
    low = np.percentile(boot_means, 100 * (alpha / 2)).round(4)
    high = np.percentile(boot_means, 100 * (1 - alpha / 2)).round(4)
    return {'mean': mean, 'std': std, 'ci_low': low, 'ci_high': high, 'n': n}


def aggregate_diversity_report(df, model, seed_col='seed_article_id', n_boot=2000):
    per_seed = per_seed_scores(df, model, seed_col=seed_col)
    div_stats = bootstrap_mean_ci(per_seed['diversity'].values, n_boot=n_boot, ci=0.95, random_state=42)
    cgi_stats = bootstrap_mean_ci(per_seed['cgi'].values, n_boot=n_boot, ci=0.95, random_state=43)
    return {
        'per_seed_df': per_seed,
        'diversity': div_stats,
        'cgi': cgi_stats
    }


# =========================
# 5. 추천 & 평가 실행 예시
# =========================
# seed articles 예시
base_articles = []
titles = [
    "배터리 자립",
    "LH투기 후폭풍, LH 타사업지 사업 차질 현실화",
    "대선 앞두고 ‘中 심판론’ 고삐 죄는 트럼프"
]

for t in titles:
    row = article[article['title'].str.contains(t)].iloc[0]
    base_articles.append({
        'context_id': row['context_id'],
        'title': row['title'],
        'context': row['context'],
        'published': row['published'],
        'source': row['source']
    })

# 추천 수행
recommend_articles = pd.DataFrame()
for base in base_articles:
    top_articles = recommend_articles_by_cosine(base, article, model)
    recommend_articles = pd.concat([recommend_articles, top_articles], ignore_index=True)

# 평가 수행
report = aggregate_diversity_report(recommend_articles, model, seed_col='base_id', n_boot=2000)
print(report)