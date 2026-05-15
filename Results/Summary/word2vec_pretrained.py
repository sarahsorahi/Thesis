"""
Persian Word2Vec Pre-trained Vector Extractor
===============================================
این کد از Word2Vec pre-trained فارسی استفاده می‌کنه
و vectors رو برای داده‌ات استخراج می‌کنه.

نصب:
    pip install gensim pandas numpy tqdm

اجرا:
    python word2vec_pretrained.py --data am_corpus_final.csv

مدل pre-trained از:
    https://fasttext.cc/docs/en/crawl-vectors.html
    فایل: cc.fa.300.vec.gz  (~2GB)
"""

import os
import gzip
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("results_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD OR LOAD PRE-TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained_w2v(model_path=None):
    """
    مدل Word2Vec pre-trained فارسی رو بارگذاری می‌کنه.
    
    گزینه ۱: فایل .vec یا .vec.gz از fasttext.cc
    گزینه ۲: فایل .bin از منابع دیگه
    """
    from gensim.models import KeyedVectors

    # اگه مدل مستقیم داده شده
    if model_path and Path(model_path).exists():
        print(f"Loading pre-trained model from: {model_path}")

        if str(model_path).endswith('.bin'):
            model = KeyedVectors.load_word2vec_format(
                model_path, binary=True
            )
        elif str(model_path).endswith('.gz'):
            print("Decompressing .gz file...")
            vec_path = str(model_path).replace('.gz', '')
            with gzip.open(model_path, 'rb') as f_in:
                with open(vec_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            model = KeyedVectors.load_word2vec_format(
                vec_path, binary=False
            )
        else:
            model = KeyedVectors.load_word2vec_format(
                model_path, binary=False
            )

        print(f"✓ Model loaded: {len(model)} words, {model.vector_size} dimensions")
        return model

    # اگه مدل نداریم، از روش جایگزین استفاده می‌کنیم
    print("No pre-trained model found.")
    print("Please download cc.fa.300.vec.gz from:")
    print("  https://fasttext.cc/docs/en/crawl-vectors.html")
    print("Then run: python word2vec_pretrained.py --data your_data.csv --w2v cc.fa.300.vec.gz")

    # fallback: train روی Wikipedia فارسی اگه داشتیم
    # یا return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACT VECTORS
# ─────────────────────────────────────────────────────────────────────────────

def extract_vectors(df, model):
    """
    برای هر جمله، vector کلمه دارای -ام رو استخراج می‌کنه.
    اگه کلمه در مدل نبود، mean of sentence words رو برمی‌گردونه.
    """
    print(f"\nExtracting vectors for {len(df)} sentences...")

    vectors      = []
    found_tokens = []
    not_found    = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Word2Vec"):
        sentence = str(row['sentence'])
        token    = str(row.get('token', row.get('lemma_guess', ''))).strip()

        vec = None

        # اول سعی کن token مستقیم پیدا کنی
        if token and token in model:
            vec = model[token]
            found_tokens.append(token)

        # اگه نبود، کلمه‌ای که به م ختم می‌شه پیدا کن
        if vec is None:
            words = sentence.split()
            for word in words:
                if word.endswith('م') and word in model:
                    vec = model[word]
                    found_tokens.append(word)
                    break

        # اگه باز هم نبود، mean of all words
        if vec is None:
            words      = sentence.split()
            valid_vecs = [model[w] for w in words if w in model]
            if valid_vecs:
                vec = np.mean(valid_vecs, axis=0)
                found_tokens.append('[sentence_mean]')
            else:
                vec = np.zeros(model.vector_size)
                found_tokens.append('[zero]')
                not_found += 1

        vectors.append(vec)

    coverage = (len(df) - not_found) / len(df) * 100
    print(f"✓ Coverage: {coverage:.1f}% ({not_found} sentences got zero vectors)")

    return np.array(vectors), found_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 3. SVM CLASSIFICATION (train/test split)
# ─────────────────────────────────────────────────────────────────────────────

def run_svm(train_vecs, test_vecs, train_labels, test_labels):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score)

    print("\n── SVM Classification: Word2Vec ──")

    le      = LabelEncoder()
    le.fit(train_labels + test_labels)
    y_train = le.transform(train_labels)
    y_test  = le.transform(test_labels)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_vecs)
    X_test  = scaler.transform(test_vecs)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(macro_f1, 4),
        "classes":   le.classes_.tolist(),
        "cm":        confusion_matrix(y_test, y_pred).tolist(),
        "report":    classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTERACTIVE VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def interactive_plot(vectors, df, method='tsne'):
    import plotly.express as px
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print(f"\n── Interactive {method.upper()}: Word2Vec ──")

    X = StandardScaler().fit_transform(vectors)

    if method == 'tsne':
        reducer = TSNE(
            n_components=2, random_state=42,
            perplexity=min(30, len(X) // 4)
        )
        X2d   = reducer.fit_transform(X)
        title = "t-SNE: Word2Vec (pre-trained)"
        xlab, ylab = "Dimension 1", "Dimension 2"
    else:
        reducer = PCA(n_components=2, random_state=42)
        X2d     = reducer.fit_transform(X)
        var     = reducer.explained_variance_ratio_
        title   = f"PCA: Word2Vec (PC1={var[0]:.1%}, PC2={var[1]:.1%})"
        xlab, ylab = f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})"

    cat_labels = {
        'verbal':     'Verbal (رفتم)',
        'possessive': 'Possessive (کتابم)',
        'copular':    'Copular (خوشحالم)',
    }

    plot_df = pd.DataFrame({
        'x':              X2d[:, 0],
        'y':              X2d[:, 1],
        'category':       df['category'].tolist(),
        'category_label': df['category'].map(cat_labels).tolist(),
        'token':          df.get('token', df.get('lemma_guess',
                          pd.Series(['?'] * len(df)))).tolist(),
        'sentence_short': df['sentence'].apply(
            lambda s: s[:80] + '...' if len(str(s)) > 80 else str(s)
        ).tolist(),
    })

    fig = px.scatter(
        plot_df,
        x='x', y='y',
        color='category_label',
        color_discrete_map={
            'Verbal (رفتم)':       '#2E75B6',
            'Possessive (کتابم)':  '#70AD47',
            'Copular (خوشحالم)':   '#ED7D31',
        },
        hover_data={
            'x': False, 'y': False,
            'category_label':  True,
            'token':           True,
            'sentence_short':  True,
        },
        labels={
            'category_label': 'Category',
            'token':          'Token (-am)',
            'sentence_short': 'Sentence',
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.75,
                    line=dict(width=0.5, color='white')),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Token: %{customdata[1]}<br>"
            "Sentence: %{customdata[2]}<br>"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        width=900, height=650,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Times New Roman", size=13),
        title=dict(font=dict(size=16, color='#1F3864'), x=0.5),
        xaxis=dict(title=xlab, showgrid=True,
                   gridcolor='#EEEEEE', zeroline=False),
        yaxis=dict(title=ylab, showgrid=True,
                   gridcolor='#EEEEEE', zeroline=False),
        legend=dict(
            title="Category",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )

    fname = OUTPUT_DIR / f"{method}_word2vec_pretrained.html"
    fig.write_html(str(fname))
    print(f"  ✓ Saved: {fname}")

    try:
        fig.write_image(
            str(OUTPUT_DIR / f"{method}_word2vec_pretrained.png"),
            width=900, height=650, scale=2
        )
    except Exception:
        print("  (PNG needs kaleido: pip install kaleido)")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to am_corpus_final.csv')
    parser.add_argument('--w2v',  type=str, default=None,
                        help='Path to pre-trained Word2Vec model (.vec, .vec.gz, or .bin)')
    args = parser.parse_args()

    print("=" * 60)
    print("Word2Vec Pre-trained: Persian -am Analysis")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    # load existing splits if available
    train_path = OUTPUT_DIR / "train_split.csv"
    test_path  = OUTPUT_DIR / "test_split.csv"

    if train_path.exists() and test_path.exists():
        print("✓ Loading existing train/test splits...")
        train_df = pd.read_csv(train_path, encoding='utf-8-sig')
        test_df  = pd.read_csv(test_path,  encoding='utf-8-sig')
    else:
        print("Creating new train/test split...")
        df = pd.read_csv(args.data, encoding='utf-8-sig')
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[df['category'].isin(
            ['verbal', 'possessive', 'copular']
        )].dropna(subset=['sentence', 'category']).reset_index(drop=True)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['category']
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
        test_df.to_csv(test_path,   index=False, encoding='utf-8-sig')

    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)} | Total: {len(full_df)}")

    # ── Load model ───────────────────────────────────────────────
    model = load_pretrained_w2v(args.w2v)
    if model is None:
        print("\nCannot proceed without a pre-trained model.")
        print("Download cc.fa.300.vec.gz and re-run with --w2v flag.")
        return

    # ── Extract vectors ──────────────────────────────────────────
    full_vecs, found_tokens = extract_vectors(full_df, model)
    train_vecs = full_vecs[:len(train_df)]
    test_vecs  = full_vecs[len(train_df):]

    # ── SVM ──────────────────────────────────────────────────────
    svm_results = run_svm(
        train_vecs, test_vecs,
        train_df['category'].tolist(),
        test_df['category'].tolist()
    )

    # ── Silhouette ───────────────────────────────────────────────
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le  = LabelEncoder()
    y   = le.fit_transform(full_df['category'].tolist())
    X   = StandardScaler().fit_transform(full_vecs)
    sil = silhouette_score(X, y, metric='cosine', random_state=42)
    print(f"\n  Silhouette score: {sil:.4f}")

    # ── Cosine similarity ─────────────────────────────────────────
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    cats      = ['verbal', 'possessive', 'copular']
    centroids = {
        c: full_vecs[np.array(full_df['category'].tolist()) == c].mean(0)
        for c in cats
    }
    cosine = {
        f"{c1}_vs_{c2}": round(float(
            cos_sim(centroids[c1].reshape(1,-1),
                    centroids[c2].reshape(1,-1))[0,0]
        ), 4)
        for c1 in cats for c2 in cats
    }
    print("\nCosine similarities:")
    for k, v in cosine.items():
        if k.split('_vs_')[0] != k.split('_vs_')[1]:
            print(f"  {k}: {v}")

    # ── Interactive visualizations ────────────────────────────────
    interactive_plot(full_vecs, full_df, 'tsne')
    interactive_plot(full_vecs, full_df, 'pca')

    # ── Save results ──────────────────────────────────────────────
    import json
    results = {
        "Word2Vec_pretrained": {
            "accuracy":   svm_results["accuracy"],
            "macro_f1":   svm_results["macro_f1"],
            "silhouette": round(float(sil), 4),
            "cosine":     cosine,
            "report":     svm_results["report"],
        }
    }

    out_path = OUTPUT_DIR / "results_word2vec_pretrained.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {out_path}")
    print("\n" + "=" * 60)
    print("Done! Open the HTML files in your browser to see")
    print("interactive visualizations with hover feature.")
    print("=" * 60)


if __name__ == "__main__":
    main()
