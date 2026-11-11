
import re

# Main skills recognized by the system
SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "sql",
    "mongodb",
    "flask",
    "django",
    "machine learning",
    "deep learning",
    "data analysis",
    "pandas",
    "numpy",
    "scikit-learn",
    "nlp",
    "communication",
    "teamwork"
]

# Alternative names and abbreviations for skills
SKILL_ALIASES = {
    "python": ["python3", "python programming"],
    "java": ["java programming", "core java"],
    "c": ["c programming"],
    "c++": ["cpp", "cplusplus", "c plus plus"],
    "html": ["html5"],
    "css": ["css3"],
    "javascript": ["ecmascript"],    "react": ["react.js", "reactjs", "react js"],
    "node.js": ["nodejs", "node js"],
    "sql": ["mysql", "structured query language"],
    "mongodb": ["mongo db"],
    "flask": ["flask framework"],
    "django": ["django framework"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "data analysis": ["data analytics", "data analyzing"],
    "pandas": [],
    "numpy": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "nlp": ["natural language processing"],
    "communication": ["communication skills", "verbal communication"],
    "teamwork": ["team work", "team collaboration", "collaboration"]
}


def clean_text(text):
    """Clean and normalize input text."""

    text = text.lower()

    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_skills(text):
    """Extract skills using canonical names and alternative keywords."""

    cleaned_text = clean_text(text)

    found_skills = []

    for skill in SKILLS:

        aliases = SKILL_ALIASES.get(skill, [])

        keywords = [skill] + aliases

        for keyword in keywords:

            # Match complete words rather than partial words
            if skill == "c":
                pattern = (
                    r"(?<![a-z0-9])"
                    + re.escape(keyword)
                    + r"(?![a-z0-9+#])"
                )
            else:
                pattern = (
                    r"(?<![a-z0-9])"
                    + re.escape(keyword)
                    + r"(?![a-z0-9])"
                )

            if re.search(pattern, cleaned_text):

                found_skills.append(skill)

                break

    return found_skills


if __name__ == "__main__":

    sample = """
    I know Python3, Java, C++, HTML5, CSS3,
    React.js, NodeJS, SQL, ML, Data Analytics,
    Pandas, NumPy, Sklearn, NLP and teamwork.
    """

    print("Cleaned Text:")
    print(clean_text(sample))

    print("\nIdentified Skills:")

    for skill in extract_skills(sample):
        print("-", skill)
