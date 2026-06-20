"""
core/search_engine.py
─────────────────────
Hybrid Information Retrieval engine for Gol D Lyric.

Algorithms
──────────
• Lyric search  : Vector Space Model (VSM) with TF-IDF weighting
                  + Cosine Similarity ranking
                  + Sastrawi stemmer (Indonesian) with NLTK stopwords fallback
• Title / Artist: Boolean Retrieval (case-insensitive substring match)

Index lifecycle
───────────────
1. build_index()  called ONCE at FastAPI startup (lifespan context)
2. The fitted TfidfVectorizer + sparse matrix live in RAM
3. Each search() call only performs a cheap transform + dot-product
"""

import json
import re
import html
import logging
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Snippet window: chars before and after the matched keyword
SNIPPET_PRE  = 40
SNIPPET_POST = 80
MAX_RESULTS  = 20

# Minimum cosine similarity score for a lyric result to appear
LYRIC_SCORE_THRESHOLD = 0.05

# ─── Indonesian stopwords (lightweight fallback if Sastrawi not installed) ──

_ID_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "adalah", "dengan",
    "untuk", "pada", "dalam", "tidak", "saya", "kamu", "dia", "kami",
    "kita", "mereka", "aku", "ku", "mu", "nya", "akan", "juga", "sudah",
    "lagi", "sudah", "bisa", "ada", "oleh", "karena", "tapi", "atau",
    "ya", "pun", "saat", "kalau", "jika", "seperti", "lebih", "sangat",
}


# ─── Preprocessing ────────────────────────────────────────────────────────────

def _try_load_sastrawi():
    """Attempt to load Sastrawi stemmer; fall back to identity function."""
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # type: ignore
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return stemmer.stem
    except ImportError:
        logger.warning("Sastrawi not found — stemming disabled. Run: pip install PySastrawi")
        return lambda text: text


_stem = _try_load_sastrawi()


def preprocess(text: str) -> str:
    """
    Normalise Indonesian lyric text for TF-IDF vectorisation.
    Steps: lowercase → strip punctuation → remove stopwords → stem
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters + spaces only
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ID_STOPWORDS and len(t) > 1]
    tokens = [_stem(t) for t in tokens]
    return " ".join(tokens)


# ─── Snippet builder ─────────────────────────────────────────────────────────

def build_snippet(lyric: str, keyword: str) -> str:
    """
    Find the first occurrence of `keyword` (case-insensitive) in `lyric`,
    extract a surrounding window, and wrap the match in a <b> tag.

    Returns a safe HTML string suitable for dangerouslySetInnerHTML after
    DOMPurify sanitisation on the frontend.
    """
    # Escape the raw lyric first to prevent any XSS from the data itself
    safe_lyric = html.escape(lyric)
    safe_keyword = re.escape(keyword)

    pattern = re.compile(safe_keyword, re.IGNORECASE)
    match = pattern.search(safe_lyric)

    if not match:
        # Fallback: return the first two lines of the lyric
        lines = safe_lyric.strip().splitlines()
        fallback = " ".join(lines[:2])
        return fallback[:SNIPPET_PRE + SNIPPET_POST] + "…"

    start, end = match.start(), match.end()

    # Determine excerpt window
    excerpt_start = max(0, start - SNIPPET_PRE)
    excerpt_end   = min(len(safe_lyric), end + SNIPPET_POST)

    prefix  = ("…" if excerpt_start > 0 else "") + safe_lyric[excerpt_start:start]
    matched = safe_lyric[start:end]
    suffix  = safe_lyric[end:excerpt_end] + ("…" if excerpt_end < len(safe_lyric) else "")

    # Wrap the matched keyword — use <b> as per TSD spec; frontend can style via CSS
    highlighted = f'<b class="highlight">{matched}</b>'
    return prefix + highlighted + suffix


def build_fallback_snippet(lyric: str) -> str:
    """Return the first ~120 chars of a lyric, HTML-escaped."""
    safe = html.escape(lyric.strip())
    lines = safe.splitlines()
    snippet = " ".join(lines[:2])
    return snippet[:120] + ("…" if len(snippet) > 120 else "")


# ─── Search Engine ────────────────────────────────────────────────────────────

class SearchEngine:
    """
    Wraps a pre-computed TF-IDF index and exposes three search methods.

    Attributes
    ──────────
    songs        : list[dict]  — raw song records loaded from JSON
    vectorizer   : TfidfVectorizer fitted on preprocessed lyrics
    tfidf_matrix : sparse (n_docs × n_terms) matrix
    """

    def __init__(self, data_path: str):
        self.data_path   = Path(data_path)
        self.songs:  list[dict] = []
        self._id_map: dict[str, dict] = {}   # uuid → song for O(1) lookup
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    # ── Index construction ───────────────────────────────────────────────────

    def build_index(self) -> None:
        """
        Load song data from JSON and fit the TF-IDF vectorizer.
        Called once at application startup.
        """
        if not self.data_path.exists():
            logger.warning(f"Data file not found at {self.data_path} — loading sample data.")
            self.songs = _sample_songs()
        else:
            with open(self.data_path, encoding="utf-8") as f:
                raw_data = json.load(f)
                # Mengecek apakah data dibungkus dalam key "lyrics" ala Kaggle
                if isinstance(raw_data, dict) and "lyrics" in raw_data:
                    self.songs = raw_data["lyrics"]
                elif isinstance(raw_data, dict) and "data" in raw_data:
                    self.songs = raw_data["data"]
                else:
                    self.songs = raw_data

        # Ensure every song has a stable UUID
        for song in self.songs:
            if "id" not in song:
                song["id"] = str(uuid.uuid4())

        # Build id → song lookup
        self._id_map = {s["id"]: s for s in self.songs}

        # Preprocess lyrics corpus
        corpus = [preprocess(s.get("lyric", "")) for s in self.songs]

        # Fit TF-IDF vectorizer (unigrams + bigrams, sublinear TF scaling)
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info(
            f"TF-IDF matrix shape: {self.tfidf_matrix.shape} "
            f"({self.tfidf_matrix.nnz} non-zero entries)"
        )

    # ── Public search methods ────────────────────────────────────────────────

    def search_by_lyric(self, query: str) -> list[dict]:
        """
        VSM cosine similarity search over the TF-IDF lyric corpus.
        Returns top-N results sorted by score descending.
        """
        if self.vectorizer is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        processed_query = preprocess(query)
        if not processed_query.strip():
            return []

        query_vec = self.vectorizer.transform([processed_query])
        scores    = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top indices sorted by score
        top_indices = np.argsort(scores)[::-1][:MAX_RESULTS]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < LYRIC_SCORE_THRESHOLD:
                break
            song = self.songs[idx]
            results.append({
                "id":      song["id"],
                "title":   song["title"],
                "artist":  song["artist"],
                "snippet": build_snippet(song.get("lyric", ""), query),
                "score":   round(score, 4),
            })

        return results

    def search_by_title(self, query: str) -> list[dict]:
        """Boolean substring match on song title (case-insensitive)."""
        q = query.lower().strip()
        results = []
        for song in self.songs:
            if q in song.get("title", "").lower():
                results.append({
                    "id":      song["id"],
                    "title":   song["title"],
                    "artist":  song["artist"],
                    "snippet": build_fallback_snippet(song.get("lyric", "")),
                    "score":   1.0,
                })
        return results[:MAX_RESULTS]

    def search_by_artist(self, query: str) -> list[dict]:
        """Boolean substring match on artist name (case-insensitive)."""
        q = query.lower().strip()
        results = []
        for song in self.songs:
            if q in song.get("artist", "").lower():
                results.append({
                    "id":      song["id"],
                    "title":   song["title"],
                    "artist":  song["artist"],
                    "snippet": build_fallback_snippet(song.get("lyric", "")),
                    "score":   1.0,
                })
        return results[:MAX_RESULTS]

    def get_song_by_id(self, song_id: str) -> Optional[dict]:
        """O(1) lookup for full song data (including lyric) by UUID."""
        return self._id_map.get(song_id)


# ─── Sample data (used when data/songs.json is absent) ──────────────────────

def _sample_songs() -> list[dict]:
    """Seed data matching the Figma screenshots for dev/demo purposes."""
    return [
        {
            "id":     "a1b2c3d4-0001",
            "title":  "Sampai Menutup Mata",
            "artist": "Acha Septriasa",
            "lyric": (
                "Embun di pagi buta\n"
                "Menebarkan bau basah\n"
                "Detik demi detik kuhitung\n"
                "Inikah saatku pergi?\n\n"
                "Oh, Tuhan, ku cinta dia\n"
                "Berikanlah aku hidup\n"
                "Takkan kusakiti dia\n"
                "Hukum aku bila terjadi\n\n"
                "Aku tak mudah untuk mencintai\n"
                "Aku tak mudah mengaku ku cinta\n"
                "Aku tak mudah mengatakan\n"
                "Aku jatuh cinta"
            ),
        },
        {
            "id":     "a1b2c3d4-0002",
            "title":  "My Heart",
            "artist": "Acha Septriasa",
            "lyric": (
                "Dengarkanlah aku sebentar\n"
                "Ada yang ingin kusampaikan\n"
                "Tentang rasa yang kusimpan\n"
                "Sudah lama dalam hatiku\n\n"
                "Kamu adalah segalanya\n"
                "Yang ada di pikiranku\n"
                "My heart only beats for you"
            ),
        },
        {
            "id":     "a1b2c3d4-0003",
            "title":  "Haruskah Kumati",
            "artist": "Ada Band",
            "lyric": (
                "Haruskah kumati demi cintamu\n"
                "Haruskah ku pergi jauh darimu\n"
                "Tidakkah kau lihat air mataku\n"
                "Mengalir deras membasahi bumi"
            ),
        },
        {
            "id":     "a1b2c3d4-0004",
            "title":  "Karena Wanita - Ingin Dimengerti",
            "artist": "Ada Band",
            "lyric": (
                "Karena wanita ingin dimengerti\n"
                "Bukan hanya dibelai dan dicintai\n"
                "Karena wanita punya sejuta mimpi\n"
                "Yang tak selalu bisa kau mengerti"
            ),
        },
        {
            "id":     "a1b2c3d4-0005",
            "title":  "Kau Auraku",
            "artist": "Ada Band",
            "lyric": (
                "Kaulah auraku\n"
                "Yang selalu menerangi jalanku\n"
                "Tanpamu hidupku gelap gulita\n"
                "Kau satu-satunya cahayaku"
            ),
        },
        {
            "id":     "a1b2c3d4-0006",
            "title":  "Langit Tujuh Bidadari",
            "artist": "Ada Band",
            "lyric": (
                "Di langit tujuh bidadari menari\n"
                "Memanggil namamu dengan merdu\n"
                "Kau seperti bidadari turun ke bumi\n"
                "Membuatku terpesona selalu"
            ),
        },
    ]