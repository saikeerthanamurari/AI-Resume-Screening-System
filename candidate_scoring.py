
def calculate_match_score(candidate_skills, required_skills):
    """Calculate the percentage of matching required skills."""

    required_set = set(required_skills)
    candidate_set = set(candidate_skills)

    if not required_set:
        return 0

    matched_skills = candidate_set.intersection(required_set)

    score = (
        len(matched_skills) / len(required_set)
    ) * 100

    return round(score, 2)


def analyze_candidate(candidate_skills, required_skills):
    """Generate a detailed candidate skill analysis."""

    candidate_set = set(candidate_skills)
    required_set = set(required_skills)

    matched = sorted(
        candidate_set.intersection(required_set)
    )

    missing = sorted(
        required_set - candidate_set
    )

    extra = sorted(
        candidate_set - required_set
    )

    score = calculate_match_score(
        candidate_skills,
        required_skills
    )

    # Generate a simple score interpretation
    if score >= 80:
        summary = "Strong skill match"
    elif score >= 60:
        summary = "Good skill match"
    elif score >= 40:
        summary = "Moderate skill match"
    else:
        summary = "Low skill match"

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "match_score": score,
        "total_required_skills": len(required_set),
        "total_matched_skills": len(matched),
        "total_missing_skills": len(missing),
        "summary": summary
    }


if __name__ == "__main__":

    candidate = [
        "python", "java", "sql",
        "html", "css", "machine learning",
        "react", "pandas"
    ]

    job_requirements = [
        "python", "sql",
        "machine learning", "flask",
        "pandas"
    ]

    result = analyze_candidate(
        candidate,
        job_requirements
    )

    print("\nCANDIDATE SKILL ANALYSIS")
    print("=" * 35)

    print("Matched Skills:", result["matched_skills"])
    print("Missing Skills:", result["missing_skills"])
    print("Extra Skills:", result["extra_skills"])

    print("\nMatching Percentage:", result["match_score"], "%")

    print("Total Required Skills:", result["total_required_skills"])
    print("Total Matched Skills:", result["total_matched_skills"])
    print("Total Missing Skills:", result["total_missing_skills"])

    print("\nAnalysis Summary:", result["summary"])
