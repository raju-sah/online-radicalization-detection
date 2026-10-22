from radpath.vocab import count_terms, log_odds_informative_dirichlet, strip_terms, top_terms


def test_specific_term_ranks_first_and_strip_removes_it():
    target = ["zorp alpha beta"] * 30 + ["alpha beta"] * 30
    contrast = ["alpha beta gamma"] * 60
    ranked = log_odds_informative_dirichlet(count_terms(target), count_terms(contrast), min_count=5)
    assert ranked[0][0] == "zorp"
    assert ranked[-1][0] == "gamma"
    assert strip_terms(["zorp alpha"], top_terms(ranked, 1)) == ["alpha"]
