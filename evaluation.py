# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

def ild_score(recommend_articles, model):
    # 텍스트 결합
    texts = (recommend_articles['title'] + " " + recommend_articles['context']).tolist()
    N = len(texts)

    if N < 2:
        return 0.0

    # 문장 임베딩
    embeddings = model.encode(texts, convert_to_tensor=False)

    # 코사인 유사도 행렬
    sim_matrix = cosine_similarity(embeddings)

    # i < j 인 쌍만 선택 (상삼각 행렬)
    upper_sim = sim_matrix[np.triu_indices(N, k=1)]

    # ILD 계산: (2 / (N*(N-1))) * Σ(1 - sim(i,j))
    ild = (2 / (N * (N - 1))) * np.sum(1 - upper_sim)

    return float(round(ild, 4))


def cgi_score(recommend_articles):
    publisher_list = recommend_articles['source'].tolist()
    print(f"CGI 디버깅: publisher_list = {publisher_list}")
    
    if len(publisher_list) < 2:
        print(f"CGI 디버깅: 기사 수 부족 ({len(publisher_list)}개)")
        return 0.0

    counts = np.array([publisher_list.count(p) for p in set(publisher_list)])
    print(f"CGI 디버깅: 출처별 카운트 = {counts}")
    
    counts = np.sort(counts)  # 오름차순 정렬
    n = len(counts)
    S_i = np.cumsum(counts)
    S_n = S_i[-1]
    
    term1 = (2 * np.sum(S_i[:-1])) / (n * S_n)
    term2 = 1 / n
    cgi = 1 - term1 - term2
    
    print(f"CGI 디버깅: n={n}, S_n={S_n}, CGI={cgi}")

    return float(round(cgi, 4))


def per_seed_scores(df, model, seed_col='seed_article_id'):
    rows = []
    for seed, group in df.groupby(seed_col):
        if len(group) < 2:
            rows.append((seed, np.nan, np.nan, len(group)))
            continue
        d = ild_score(group, model)
        c = cgi_score(group)
        rows.append((seed, d, c, len(group)))
    return pd.DataFrame(rows, columns=[seed_col, 'ild', 'cgi', 'n_recommendations'])


def bootstrap_mean_ci(values, n_boot=1000, ci=0.95, random_state=None):
    vals = np.array(values)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return {'mean': 0.0, 'std': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'n': 0}
    rng = np.random.RandomState(random_state)
    boot_means = [np.mean(rng.choice(vals, size=n, replace=True)) for _ in range(n_boot)]
    mean = float(round(np.mean(vals), 4))
    std = float(round(np.std(vals, ddof=1), 4)) if n > 1 else 0.0
    alpha = 1 - ci
    low = float(round(np.percentile(boot_means, 100 * (alpha / 2)), 4))
    high = float(round(np.percentile(boot_means, 100 * (1 - alpha / 2)), 4))
    return {'mean': mean, 'std': std, 'ci_low': low, 'ci_high': high, 'n': n}


def aggregate_diversity_report(df, model, seed_col='seed_article_id', n_boot=2000):
    per_seed = per_seed_scores(df, model, seed_col=seed_col)
    ild_stats = bootstrap_mean_ci(per_seed['ild'].values, n_boot=n_boot, ci=0.95, random_state=42)
    cgi_stats = bootstrap_mean_ci(per_seed['cgi'].values, n_boot=n_boot, ci=0.95, random_state=43)
    return {
        'per_seed_df': per_seed,
        'ild': ild_stats,
        'cgi': cgi_stats
    }