
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_text_similarity(resume_text, job_description):
    """Compare resume text with a job description using TF-IDF."""

    if not resume_text.strip() or not job_description.strip():
        return 0.0

    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


if __name__ == "__main__":

    resume = """
    Python developer with knowledge of Java,
    SQL, machine learning, Flask and data analysis.
    """

    job_description = """
    Looking for a Python developer with experience
    in machine learning, SQL, Flask and data analysis.
    """

    score = calculate_text_similarity(
        resume,
        job_description
    )

    print("Resume and Job Description Analysis")
    print("-----------------------------------")
    print("TF-IDF Similarity Score:", score, "%")
