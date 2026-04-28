"""
Synthetic annotation dataset generator.
Simulates realistic annotator behavior with configurable noise and bias.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

# ─── ANNOTATION TASK DEFINITIONS ──────────────────────────────────────────────

ANNOTATION_TASKS = {
    "Mood Classification": {
        "description": "Classify the primary mood/emotion conveyed by a music track.",
        "labels": ["Happy", "Sad", "Energetic", "Calm", "Angry", "Romantic"],
        "guidelines": [
            "🎵 <b>Happy:</b> Upbeat tempo (>120 BPM), major key, positive lyrical themes. When in doubt between Happy and Energetic, use tempo as tiebreaker (>130 = Energetic).",
            "😔 <b>Sad:</b> Minor key, slow tempo (<80 BPM), themes of loss, longing, or melancholy. Acoustic instruments often signal Sad.",
            "⚡ <b>Energetic:</b> High tempo (>130 BPM), driving rhythm, club/workout context. Electronic/EDM defaults to Energetic unless tempo <100.",
            "🌿 <b>Calm:</b> Slow tempo (<90 BPM), major key, minimal percussion. Ambient, lo-fi, and classical often fall here.",
            "😤 <b>Angry:</b> Distorted guitar, aggressive vocals, fast tempo with harsh timbre. Metal and hardcore typically qualify.",
            "❤️ <b>Romantic:</b> Soft vocals, moderate tempo (70–110 BPM), love-themed lyrics. R&B and soul are common markers.",
            "⚠️ <b>Edge case rule:</b> If two moods apply equally, flag for review. Do NOT split — choose the dominant mood."
        ]
    },
    "Content Suitability (Age Rating)": {
        "description": "Rate content suitability for age-based playlist filtering.",
        "labels": ["All Ages", "Teen+", "Mature", "Explicit"],
        "guidelines": [
            "👶 <b>All Ages:</b> No profanity, no adult themes, no violence. Children's music, classical, and most pop without explicit lyrics.",
            "🧑 <b>Teen+:</b> Mild themes (heartbreak, mild language), no graphic content. Most mainstream pop and hip-hop without F-bombs.",
            "🔞 <b>Mature:</b> Moderate adult themes, implied sexual content, infrequent strong language. Must be clearly non-explicit per platform rules.",
            "⛔ <b>Explicit:</b> Profanity per platform guidelines (F-word, N-word, etc.), graphic violence, or overt sexual content. Use Spotify's explicit tag as primary signal.",
            "⚠️ <b>Edge case rule:</b> When lyrics are ambiguous (artistic/metaphorical), default to the less restrictive rating. Escalate to Content Policy if unsure."
        ]
    },
    "Genre Tagging": {
        "description": "Assign a primary genre tag to enable accurate content discovery.",
        "labels": ["Pop", "Hip-Hop", "Rock", "Electronic", "R&B", "Jazz/Blues"],
        "guidelines": [
            "🎤 <b>Pop:</b> Mainstream commercial appeal, broad audience, hook-driven structure. If genre is ambiguous but clearly mainstream, default to Pop.",
            "🎤 <b>Hip-Hop:</b> Rap vocals (spoken-word over beat), trap rhythms, boom-bap. Sub-genres (trap, drill) all map to Hip-Hop.",
            "🎸 <b>Rock:</b> Guitar-driven, band format. Includes indie, alternative, metal, and classic rock. Electronic-rock hybrids go to Electronic.",
            "🎛️ <b>Electronic:</b> Produced primarily with synthesizers/DAW. Includes EDM, house, techno, ambient, and lo-fi. Vocals OK.",
            "🎵 <b>R&B:</b> Soul-influenced vocals, groove-oriented rhythm, often romantic themes. Distinguish from Hip-Hop by primary vocal style (sung vs rapped).",
            "🎷 <b>Jazz/Blues:</b> Improvisational elements, 12-bar blues structure, or jazz harmony. Includes neo-soul and jazz-influenced fusion.",
            "⚠️ <b>Edge case rule:</b> Multi-genre tracks: assign the genre that dominates the production (not the artist's typical genre)."
        ]
    },
    "Podcast Topic Classification": {
        "description": "Classify podcast episode primary topic for content discovery.",
        "labels": ["News", "Comedy", "True Crime", "Education", "Health/Wellness", "Business"],
        "guidelines": [
            "📰 <b>News:</b> Current events, journalism, political commentary. Must reference real, recent events as primary focus.",
            "😂 <b>Comedy:</b> Stand-up, sketch, satirical content where humor is the primary purpose (not incidental).",
            "🔍 <b>True Crime:</b> Investigation of real crimes, criminal psychology, legal proceedings. Fictional crime = Education.",
            "📚 <b>Education:</b> Instructional, explainer, or academic content. Includes history, science, language learning, and skills.",
            "💪 <b>Health/Wellness:</b> Mental health, fitness, nutrition, medical topics. Must be primarily health-focused, not just health-adjacent.",
            "💼 <b>Business:</b> Entrepreneurship, finance, career, marketing, and startup content. Personal finance straddles Health — use primary theme.",
            "⚠️ <b>Edge case rule:</b> Multi-topic episodes should be classified by the primary segment (>50% of runtime)."
        ]
    },
    "Audiobook Genre": {
        "description": "Classify audiobook primary genre for catalog organization.",
        "labels": ["Fiction", "Non-Fiction", "Self-Help", "Thriller/Mystery", "Sci-Fi/Fantasy", "Biography"],
        "guidelines": [
            "📖 <b>Fiction:</b> Literary fiction, contemporary, romance, and other invented narratives not covered by sub-genres below.",
            "📊 <b>Non-Fiction:</b> Factual, research-based works not primarily instructional. History, sociology, and journalism.",
            "🌱 <b>Self-Help:</b> Instructional content aimed at personal improvement, productivity, or mindset change. Includes business how-to.",
            "🔎 <b>Thriller/Mystery:</b> Suspense-driven plot, crime-solving, or psychological tension as primary narrative engine.",
            "🚀 <b>Sci-Fi/Fantasy:</b> Speculative fiction with science-based or magical world-building as a core element.",
            "👤 <b>Biography:</b> Account of a real person's life (autobiography, memoir, or third-person biography).",
            "⚠️ <b>Edge case rule:</b> Narrative non-fiction (e.g., historical true crime) goes to Thriller/Mystery if suspense-driven, else Non-Fiction."
        ]
    }
}

# ─── TRACK DATA ────────────────────────────────────────────────────────────────

TRACK_DATA = {
    "Mood Classification": [
        ("Blinding Lights", "The Weeknd", "Pop", "Energetic"),
        ("Someone Like You", "Adele", "Pop", "Sad"),
        ("Happy", "Pharrell Williams", "Pop", "Happy"),
        ("Lose Yourself", "Eminem", "Hip-Hop", "Energetic"),
        ("Stay With Me", "Sam Smith", "Pop", "Sad"),
        ("Shape of You", "Ed Sheeran", "Pop", "Happy"),
        ("All of Me", "John Legend", "R&B", "Romantic"),
        ("Closer", "Chainsmokers", "Electronic", "Happy"),
        ("Heathens", "Twenty One Pilots", "Rock", "Angry"),
        ("Perfect", "Ed Sheeran", "Pop", "Romantic"),
        ("God's Plan", "Drake", "Hip-Hop", "Calm"),
        ("Seven Nation Army", "White Stripes", "Rock", "Angry"),
        ("Weightless", "Marconi Union", "Electronic", "Calm"),
        ("Cry Me a River", "Justin Timberlake", "R&B", "Sad"),
        ("Take Five", "Dave Brubeck", "Jazz/Blues", "Calm"),
        ("Thunderstruck", "AC/DC", "Rock", "Energetic"),
        ("Hello", "Adele", "Pop", "Sad"),
        ("Love Story", "Taylor Swift", "Pop", "Romantic"),
        ("Sicko Mode", "Travis Scott", "Hip-Hop", "Energetic"),
        ("Clair de Lune", "Debussy", "Classical", "Calm"),
    ],
    "Content Suitability (Age Rating)": [
        ("Baby Shark", "Pinkfong", "Children's", "All Ages"),
        ("WAP", "Cardi B", "Hip-Hop", "Explicit"),
        ("Yellow", "Coldplay", "Rock", "All Ages"),
        ("HUMBLE.", "Kendrick Lamar", "Hip-Hop", "Explicit"),
        ("Shake It Off", "Taylor Swift", "Pop", "All Ages"),
        ("Psychosocial", "Slipknot", "Metal", "Mature"),
        ("Sunflower", "Post Malone", "Pop", "Teen+"),
        ("B*tch Better Have My Money", "Rihanna", "Pop", "Explicit"),
        ("Can't Stop the Feeling", "Justin Timberlake", "Pop", "All Ages"),
        ("Rockstar", "Post Malone", "Hip-Hop", "Explicit"),
        ("Let It Go", "Idina Menzel", "Pop", "All Ages"),
        ("Old Town Road", "Lil Nas X", "Country/Rap", "Teen+"),
        ("Bohemian Rhapsody", "Queen", "Rock", "Teen+"),
        ("Money In The Grave", "Drake", "Hip-Hop", "Explicit"),
        ("Thriller", "Michael Jackson", "Pop", "Teen+"),
    ],
    "Genre Tagging": [
        ("Blinding Lights", "The Weeknd", "Pop", "Pop"),
        ("God's Plan", "Drake", "Hip-Hop", "Hip-Hop"),
        ("Smells Like Teen Spirit", "Nirvana", "Rock", "Rock"),
        ("One More Time", "Daft Punk", "Electronic", "Electronic"),
        ("No Scrubs", "TLC", "R&B", "R&B"),
        ("So What", "Miles Davis", "Jazz", "Jazz/Blues"),
        ("Levitating", "Dua Lipa", "Pop", "Pop"),
        ("SICKO MODE", "Travis Scott", "Hip-Hop", "Hip-Hop"),
        ("Back in Black", "AC/DC", "Rock", "Rock"),
        ("Strobe", "deadmau5", "Electronic", "Electronic"),
        ("Crazy in Love", "Beyoncé", "R&B", "R&B"),
        ("Blue in Green", "Bill Evans", "Jazz", "Jazz/Blues"),
    ],
    "Podcast Topic Classification": [
        ("The Daily - Election Analysis", "NYT", "News", "News"),
        ("My Favorite Murder", "Karen/Georgia", "True Crime", "True Crime"),
        ("Conan O'Brien Needs a Friend", "Conan O'Brien", "Comedy", "Comedy"),
        ("How I Built This", "NPR", "Business", "Business"),
        ("Stuff You Should Know", "iHeart", "Education", "Education"),
        ("The Diary of a CEO", "Steven Bartlett", "Business", "Business"),
        ("Crime Junkie", "audiochuck", "True Crime", "True Crime"),
        ("Serial", "This American Life", "True Crime", "True Crime"),
        ("Planet Money", "NPR", "Education", "Business"),
        ("Your Brain on Facts", "Moxie LaBouche", "Education", "Education"),
        ("On Being", "Krista Tippett", "Health/Wellness", "Health/Wellness"),
        ("SmartLess", "Bateman/Hayes/Arnett", "Comedy", "Comedy"),
    ],
    "Audiobook Genre": [
        ("Atomic Habits", "James Clear", "Self-Help", "Self-Help"),
        ("Gone Girl", "Gillian Flynn", "Thriller/Mystery", "Thriller/Mystery"),
        ("Educated", "Tara Westover", "Biography", "Biography"),
        ("Dune", "Frank Herbert", "Sci-Fi/Fantasy", "Sci-Fi/Fantasy"),
        ("Sapiens", "Yuval Noah Harari", "Non-Fiction", "Non-Fiction"),
        ("The Midnight Library", "Matt Haig", "Fiction", "Fiction"),
        ("Becoming", "Michelle Obama", "Biography", "Biography"),
        ("The Hitchhiker's Guide", "Douglas Adams", "Sci-Fi/Fantasy", "Sci-Fi/Fantasy"),
        ("Thinking Fast and Slow", "Daniel Kahneman", "Non-Fiction", "Non-Fiction"),
        ("The Silent Patient", "Alex Michaelides", "Thriller/Mystery", "Thriller/Mystery"),
        ("The Alchemist", "Paulo Coelho", "Fiction", "Fiction"),
        ("Can't Hurt Me", "David Goggins", "Self-Help", "Self-Help"),
    ]
}


def _get_default_tracks(task_name):
    return TRACK_DATA.get(task_name, TRACK_DATA["Mood Classification"])


def _add_annotator_noise(true_label: str, labels: List[str], noise_level: float, rng: np.random.Generator) -> str:
    """Simulate annotator with given noise level."""
    if rng.random() < noise_level:
        # Choose a random label (possibly same)
        return rng.choice(labels)
    # With some probability, choose a "confusable" adjacent label
    if rng.random() < noise_level * 0.5:
        idx = labels.index(true_label)
        # Adjacent labels
        adjacent = []
        if idx > 0:
            adjacent.append(labels[idx - 1])
        if idx < len(labels) - 1:
            adjacent.append(labels[idx + 1])
        if adjacent:
            return rng.choice(adjacent)
    return true_label


def generate_annotation_dataset(
    task_name: str,
    n_tracks: int,
    n_annotators: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a realistic synthetic annotation dataset.
    Each row is a track with labels from N annotators.
    """
    rng = np.random.default_rng(seed)
    task_config = ANNOTATION_TASKS[task_name]
    labels = task_config["labels"]

    base_tracks = _get_default_tracks(task_name)

    # Build track pool by cycling/extending base data
    rows = []
    genres_pool = list(set(t[2] for t in base_tracks))

    # Annotator profiles: different noise levels per annotator
    annotator_noise = [
        rng.uniform(0.05, 0.15) for _ in range(n_annotators)
    ]
    # One "noisy" annotator
    if n_annotators >= 3:
        annotator_noise[-1] = rng.uniform(0.25, 0.45)

    for i in range(n_tracks):
        base = base_tracks[i % len(base_tracks)]
        track_name = base[0]
        artist = base[1]
        genre = base[2]
        true_label = base[3] if base[3] in labels else rng.choice(labels)

        # Add track index to name to create variety
        if i >= len(base_tracks):
            track_name = f"{track_name} (v{i // len(base_tracks) + 1})"

        # Audio features (simulated)
        energy = float(rng.uniform(0.2, 0.99))
        danceability = float(rng.uniform(0.2, 0.99))
        valence = float(rng.uniform(0.1, 0.99))
        tempo = float(rng.uniform(60, 180))
        acousticness = float(rng.uniform(0.01, 0.95))

        row = {
            "track_id": f"TRK{i:04d}",
            "track_name": track_name,
            "artist": artist,
            "genre": genre,
            "energy": round(energy, 3),
            "danceability": round(danceability, 3),
            "valence": round(valence, 3),
            "tempo": round(tempo, 1),
            "acousticness": round(acousticness, 3),
        }

        # Generate annotator labels
        for j in range(n_annotators):
            ann_label = _add_annotator_noise(
                true_label, labels, annotator_noise[j], rng
            )
            row[f"annotator_{j+1}"] = ann_label

        rows.append(row)

    return pd.DataFrame(rows)
