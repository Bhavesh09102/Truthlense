import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="TruthLense/1.0 (educational project)"
)

def get_wikipedia_page(title):
    page = wiki.page(title)
    if not page.exists():
        return None
    return page

def retrieve_evidence(entities, max_sentences=3):
    evidence = []

    for entity in entities:
        page = get_wikipedia_page(entity)
        if page is None:
            continue

        sentences = page.summary.split(". ")
        for sent in sentences[:max_sentences]:
            evidence.append(sent)

    return evidence
