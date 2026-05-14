"""
Persian -am Embedding Analysis Pipeline (Fixed Version)
=========================================================
تغییرات نسبت به نسخه قبل:
1. train/test split واقعی (80/20)
2. فقط pre-trained models (FastText + ParsBERT)
3. Interactive visualization با Plotly (hover feature)
4. SVM فقط روی train fit می‌شه، روی test evaluate می‌شه

نصب:
    pip install pandas numpy scikit-learn plotly gensim transformers torch tqdm

اجرا:
    python analyze_am_v2.py --data am_corpus_final.csv --ft cc.fa.300.bin
"""

import os
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results_v2")
OUTPUT_DIR.mkdir(exist_ok=True)

CATEGORY_COLORS = {
    "verbal":     "#2E75B6",
    "possessive": "#70AD47",
    "copular":    "#ED7D31",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND SPLIT DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(csv_path, test_size=0.2, random_state=42):
    """
    داده رو بارگذاری و split می‌کنه.
    مهم: split قبل از هر چیز دیگه‌ای انجام می‌شه.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize categories
    cat_map = {
        'verb_1sg': 'verbal', 'v_pre_nc_1': 'verbal',
        'noun_poss_1sg': 'possessive', 'n_sing_com_paff_1': 'possessive',
        'adj_sim': 'copular', 'v_pre_adjc_1': 'copular',
    }
    if 'category' in df.columns:
        df['category'] = df['category'].str.strip().str.lower().map(
            lambda x: cat_map.get(x, x)
        )

    df = df[df['category'].isin(['verbal', 'possessive', 'copular'])].copy()
    df = df.dropna(subset=['sentence', 'category']).reset_index(drop=True)

    print(f"\n✓ Total dataset: {len(df)} sentences")
    print(df['category'].value_counts().to_string())

    # stratified split — ensures each category is represented in both sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"\n✓ Train set: {len(train_df)} sentences")
    print(train_df['category'].value_counts().to_string())
    print(f"\n✓ Test set: {len(test_df)} sentences")
    print(test_df['category'].value_counts().to_string())

    # save splits for reference
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv(OUTPUT_DIR  / "test_split.csv",  index=False, encoding='utf-8-sig')

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FASTTEXT VECTORS (pre-trained only)
# ─────────────────────────────────────────────────────────────────────────────

def get_fasttext_vectors(df, model_path):
    """
    FastText pre-trained vectors.
    مدل از: https://fasttext.cc/docs/en/crawl-vectors.html
    فایل: cc.fa.300.bin
    """
    print("\n── FastText (pre-trained) ──")

    try:
        import fasttext
        ft = fasttext.load_model(str(model_path))
        print(f"✓ Model loaded: {model_path}")

        vectors = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="FastText"):
            sentence = str(row['sentence'])
            token    = str(row.get('token', '')).strip()

            # اول token مستقیم
            if token and token.endswith('م'):
                vec = ft.get_word_vector(token)
            else:
                # mean of sentence words
                words = sentence.split()
                vecs  = [ft.get_word_vector(w) for w in words if w.strip()]
                vec   = np.mean(vecs, axis=0) if vecs else np.zeros(300)

            vectors.append(vec)

        return np.array(vectors)

    except ImportError:
        print("fasttext not installed. Trying gensim...")
        try:
            from gensim.models import FastText as GensimFT
            from gensim.models import KeyedVectors
            # try loading as .vec format
            model = KeyedVectors.load_word2vec_format(str(model_path))
            vectors = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                token = str(row.get('token', '')).strip()
                if token in model:
                    vec = model[token]
                else:
                    words = str(row['sentence']).split()
                    valid = [model[w] for w in words if w in model]
                    vec   = np.mean(valid, axis=0) if valid else np.zeros(300)
                vectors.append(vec)
            return np.array(vectors)
        except Exception as e:
            print(f"Error: {e}")
            print("Using random vectors as placeholder — install fasttext!")
            return np.random.randn(len(df), 300)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PARSBERT VECTORS (pre-trained only)
# ─────────────────────────────────────────────────────────────────────────────

def get_parsbert_vectors(df, model_name="HooshvareLab/bert-fa-base-uncased"):
    """
    ParsBERT contextual embeddings.
    آخرین hidden layer برای token دارای -ام.
    """
    print(f"\n── ParsBERT ({model_name}) ──")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
        model.eval()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model  = model.to(device)
        print(f"  Device: {device}")

        vectors = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ParsBERT"):
            sentence = str(row['sentence'])

            with torch.no_grad():
                inputs = tokenizer(
                    sentence,
                    return_tensors='pt',
                    truncation=True,
                    max_length=128,
                    padding=True
                ).to(device)

                outputs = model(**inputs)
                hidden  = outputs.last_hidden_state[0]  # [seq_len, 768]

                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

                # find -am token
                am_indices = [
                    i for i, tok in enumerate(tokens)
                    if tok and (tok.endswith('م') or '##م' in tok or tok == 'ام')
                ]

                if am_indices:
                    vec = hidden[am_indices].mean(0).cpu().numpy()
                else:
                    vec = hidden[0].cpu().numpy()  # fallback: [CLS]

            vectors.append(vec)

        print(f"✓ {len(vectors)} ParsBERT vectors extracted")
        return np.array(vectors)

    except Exception as e:
        print(f"ParsBERT error: {e}")
        print("Using random vectors as placeholder!")
        return np.random.randn(len(df), 768)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SVM — TRAIN ON TRAIN, EVALUATE ON TEST
# ─────────────────────────────────────────────────────────────────────────────

def svm_train_test(train_vecs, test_vecs, train_labels, test_labels, model_name):
    """
    SVM: fit روی train، evaluate روی test.
    این methodologically درسته.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  accuracy_score, f1_score)

    print(f"\n── SVM: {model_name} ──")

    le = LabelEncoder()
    le.fit(train_labels + test_labels)

    y_train = le.transform(train_labels)
    y_test  = le.transform(test_labels)

    # scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_vecs)
    X_test  = scaler.transform(test_vecs)

    # fit on train only
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)

    # evaluate on test only
    y_pred = clf.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    return {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(macro_f1, 4),
        "report":    report,
        "y_test":    y_test.tolist(),
        "y_pred":    y_pred.tolist(),
        "classes":   le.classes_.tolist(),
        "cm":        confusion_matrix(y_test, y_pred).tolist()
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. INTERACTIVE VISUALIZATION WITH PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def interactive_plot(vectors, df, model_name, method='tsne'):
    """
    Interactive t-SNE یا PCA با Plotly.
    Hover نشون می‌ده: جمله کامل، token، category.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    print(f"\n── Interactive {method.upper()}: {model_name} ──")

    X = StandardScaler().fit_transform(vectors)

    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(X) // 4),
            max_iter=1000
        )
        X2d   = reducer.fit_transform(X)
        title = f"t-SNE: {model_name}"
        xlab, ylab = "Dimension 1", "Dimension 2"
    else:
        reducer = PCA(n_components=2, random_state=42)
        X2d     = reducer.fit_transform(X)
        var     = reducer.explained_variance_ratio_
        title   = f"PCA: {model_name} (PC1={var[0]:.1%}, PC2={var[1]:.1%})"
        xlab, ylab = f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})"

    # build dataframe for plotly
    plot_df = pd.DataFrame({
        'x':        X2d[:, 0],
        'y':        X2d[:, 1],
        'category': df['category'].tolist(),
        'token':    df.get('token', df.get('lemma_guess', pd.Series(['?'] * len(df)))).tolist(),
        'sentence': df['sentence'].tolist(),
    })

    # truncate long sentences for hover
    plot_df['sentence_short'] = plot_df['sentence'].apply(
        lambda s: s[:80] + '...' if len(str(s)) > 80 else str(s)
    )

    # category labels with Farsi
    cat_labels = {
        'verbal':     'Verbal (رفتم)',
        'possessive': 'Possessive (کتابم)',
        'copular':    'Copular (خوشحالم)',
    }
    plot_df['category_label'] = plot_df['category'].map(cat_labels)

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
            'category_label': True,
            'token': True,
            'sentence_short': True,
        },
        labels={
            'category_label': 'Category',
            'token':          'Token (-am)',
            'sentence_short': 'Sentence',
        },
        title=title,
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color='white')),
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
        xaxis=dict(title=xlab, showgrid=True, gridcolor='#EEEEEE', zeroline=False),
        yaxis=dict(title=ylab, showgrid=True, gridcolor='#EEEEEE', zeroline=False),
        legend=dict(
            title="Category",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#CCCCCC',
            borderwidth=1
        )
    )

    # save as interactive HTML
    fname = OUTPUT_DIR / f"{method}_{model_name.lower().replace(' ','_')}.html"
    fig.write_html(str(fname))
    print(f"  ✓ Saved interactive: {fname}")

    # also save as static PNG
    try:
        png_fname = OUTPUT_DIR / f"{method}_{model_name.lower().replace(' ','_')}.png"
        fig.write_image(str(png_fname), width=900, height=650, scale=2)
        print(f"  ✓ Saved static:      {png_fname}")
    except Exception:
        print("  (PNG export needs kaleido: pip install kaleido)")

    return fig


def interactive_confusion(cm, classes, model_name):
    """
    Interactive confusion matrix با Plotly.
    """
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=classes, y=classes,
        color_continuous_scale='Blues',
        title=f"Confusion Matrix: {model_name}",
        text_auto=True,
    )
    fig.update_layout(
        width=500, height=450,
        font=dict(family="Times New Roman", size=13),
        title=dict(font=dict(size=15, color='#1F3864'), x=0.5),
    )
    fname = OUTPUT_DIR / f"confusion_{model_name.lower().replace(' ','_')}.html"
    fig.write_html(str(fname))
    print(f"  ✓ Saved confusion matrix: {fname}")
    return fig


def interactive_cosine(cosine_dict, model_name, categories=['verbal','possessive','copular']):
    """
    Interactive cosine similarity heatmap.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(categories)
    matrix = np.zeros((n, n))
    for i, c1 in enumerate(categories):
        for j, c2 in enumerate(categories):
            key = f"{c1}_vs_{c2}"
            matrix[i, j] = cosine_dict.get(key, 0)

    labels = ['Verbal\n(رفتم)', 'Possessive\n(کتابم)', 'Copular\n(خوشحالم)']

    fig = px.imshow(
        matrix,
        labels=dict(x="Category", y="Category", color="Cosine Similarity"),
        x=labels, y=labels,
        color_continuous_scale='Blues',
        zmin=0.7, zmax=1.0,
        title=f"Cosine Similarity: {model_name}",
        text_auto='.3f',
    )
    fig.update_layout(
        width=500, height=450,
        font=dict(family="Times New Roman", size=13),
        title=dict(font=dict(size=15, color='#1F3864'), x=0.5),
    )
    fname = OUTPUT_DIR / f"cosine_{model_name.lower().replace(' ','_')}.html"
    fig.write_html(str(fname))
    print(f"  ✓ Saved cosine heatmap: {fname}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_cosine(vectors, labels):
    from sklearn.metrics.pairwise import cosine_similarity

    cats = ['verbal', 'possessive', 'copular']
    centroids = {}
    for cat in cats:
        mask = np.array(labels) == cat
        if mask.sum() > 0:
            centroids[cat] = vectors[mask].mean(axis=0)

    result = {}
    for c1 in cats:
        for c2 in cats:
            if c1 in centroids and c2 in centroids:
                sim = cosine_similarity(
                    centroids[c1].reshape(1, -1),
                    centroids[c2].reshape(1, -1)
                )[0, 0]
                result[f"{c1}_vs_{c2}"] = round(float(sim), 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. SILHOUETTE
# ─────────────────────────────────────────────────────────────────────────────

def silhouette(vectors, labels, model_name):
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    X  = StandardScaler().fit_transform(vectors)

    if len(set(labels)) < 2:
        return None

    score = silhouette_score(X, y, metric='cosine', random_state=42)
    print(f"  Silhouette ({model_name}): {score:.4f}")
    return round(float(score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results):
    path = OUTPUT_DIR / "results_summary_v2.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Full results: {path}")

    rows = []
    for model, m in results.items():
        rows.append({
            "Model":       model,
            "Accuracy":    m.get("svm", {}).get("accuracy", "—"),
            "Macro-F1":    m.get("svm", {}).get("macro_f1", "—"),
            "Silhouette":  m.get("silhouette", "—"),
            "V–P sim.":    m.get("cosine", {}).get("verbal_vs_possessive", "—"),
            "V–C sim.":    m.get("cosine", {}).get("verbal_vs_copular", "—"),
            "P–C sim.":    m.get("cosine", {}).get("possessive_vs_copular", "—"),
        })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "results_summary_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Summary table: {csv_path}")
    print("\n" + df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',           type=str, required=True)
    parser.add_argument('--ft',             type=str, default=None,
                        help='Path to cc.fa.300.bin')
    parser.add_argument('--skip_parsbert',  action='store_true')
    parser.add_argument('--test_size',      type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("Persian -am Embedding Analysis (v2 — Fixed)")
    print("=" * 60)

    # ── Step 1: Load and split ───────────────────────────────────
    train_df, test_df = load_and_split(args.data, test_size=args.test_size)

    # full df for visualization (all data)
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    train_labels = train_df['category'].tolist()
    test_labels  = test_df['category'].tolist()
    full_labels  = full_df['category'].tolist()

    results = {}

    # ── Step 2: FastText ─────────────────────────────────────────
    if args.ft:
        print("\n" + "="*40)
        print("MODEL 1: FastText (pre-trained)")
        print("="*40)

        # extract vectors for ALL data
        ft_full  = get_fasttext_vectors(full_df,  args.ft)
        ft_train = ft_full[:len(train_df)]
        ft_test  = ft_full[len(train_df):]

        # cosine on full data
        cosine_ft = compute_cosine(ft_full, full_labels)

        # SVM: train on train, test on test
        svm_ft = svm_train_test(
            ft_train, ft_test,
            train_labels, test_labels,
            "FastText"
        )

        # silhouette on full
        sil_ft = silhouette(ft_full, full_labels, "FastText")

        # interactive visualizations (full data)
        interactive_plot(ft_full, full_df, "FastText", 'tsne')
        interactive_plot(ft_full, full_df, "FastText", 'pca')
        interactive_cosine(cosine_ft, "FastText")
        interactive_confusion(
            np.array(svm_ft['cm']),
            svm_ft['classes'],
            "FastText"
        )

        results["FastText"] = {
            "cosine":     cosine_ft,
            "svm":        {k: v for k, v in svm_ft.items() if k not in ['y_test','y_pred','cm']},
            "silhouette": sil_ft,
        }
    else:
        print("\nFastText model not provided (--ft). Skipping.")

    # ── Step 3: ParsBERT ─────────────────────────────────────────
    if not args.skip_parsbert:
        print("\n" + "="*40)
        print("MODEL 2: ParsBERT (pre-trained)")
        print("="*40)

        pb_full  = get_parsbert_vectors(full_df)
        pb_train = pb_full[:len(train_df)]
        pb_test  = pb_full[len(train_df):]

        cosine_pb = compute_cosine(pb_full, full_labels)

        svm_pb = svm_train_test(
            pb_train, pb_test,
            train_labels, test_labels,
            "ParsBERT"
        )

        sil_pb = silhouette(pb_full, full_labels, "ParsBERT")

        interactive_plot(pb_full, full_df, "ParsBERT", 'tsne')
        interactive_plot(pb_full, full_df, "ParsBERT", 'pca')
        interactive_cosine(cosine_pb, "ParsBERT")
        interactive_confusion(
            np.array(svm_pb['cm']),
            svm_pb['classes'],
            "ParsBERT"
        )

        results["ParsBERT"] = {
            "cosine":     cosine_pb,
            "svm":        {k: v for k, v in svm_pb.items() if k not in ['y_test','y_pred','cm']},
            "silhouette": sil_pb,
        }

    # ── Save results ─────────────────────────────────────────────
    save_report(results)

    print("\n" + "=" * 60)
    print("Done! Check the 'results_v2/' folder.")
    print("\nHTML files (interactive — open in browser):")
    for f in sorted(OUTPUT_DIR.glob("*.html")):
        print(f"  {f.name}")
    print("\nCSV/JSON results:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
