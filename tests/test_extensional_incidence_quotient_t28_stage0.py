from collections import Counter

from experiments.extensional_incidence_quotient_t28_stage0 import (
    CorpusDocument,
    canonical_pattern,
    connected_components,
    enumerate_edges,
    inverted_graph,
    nonoverlapping_exact_occurrences,
    set_intersection_graph,
    split_sentences,
    tokenize,
)


def document(index: int, title: str, text: str) -> CorpusDocument:
    return CorpusDocument(
        original_index=index,
        title=title,
        title_tokens=tokenize(title),
        sentences=split_sentences(tokenize(text)),
    )


def test_tokenizer_and_sentence_split_are_frozen():
    tokens = tokenize("Alpha-beta is real! Yes?")
    assert tokens == ("alpha", "-", "beta", "is", "real", "!", "yes", "?")
    assert split_sentences(tokens) == (
        ("alpha", "-", "beta", "is", "real", "!"),
        ("yes", "?"),
    )


def test_exact_title_occurrences_are_nonoverlapping():
    tokens = tokenize("Alpha Beta met Alpha Beta.")
    needle = tokenize("Alpha Beta")
    assert nonoverlapping_exact_occurrences(tokens, needle) == ((0, 2), (3, 5))


def test_canonical_pattern_replaces_subject_and_value():
    tokens = tokenize("Alpha is a painter.")
    pattern = canonical_pattern(tokens, ((0, 1),), (3, 4))
    assert pattern == ("<S>", "is", "a", "<V>", ".")


def test_edge_enumerator_respects_raw_frequency_and_regenerates():
    documents = (
        document(0, "Alpha", "Alpha is a painter. Alpha worked as painter."),
        document(1, "Beta", "Beta is a painter."),
        document(2, "Gamma", "Gamma is a painter."),
        document(3, "Delta", "Delta is a painter."),
        document(4, "Epsilon", "Epsilon is a painter."),
        document(5, "Zeta", "Zeta is a painter."),
        document(6, "Eta", "Eta is a painter."),
        document(7, "Theta", "Theta is a painter."),
        document(8, "Iota", "Iota is a painter."),
        document(9, "Kappa", "Kappa is a painter."),
        document(10, "Lambda", "Lambda is a sculptor."),
        document(11, "Mu", "Mu is a sculptor."),
        document(12, "Nu", "Nu is a sculptor."),
        document(13, "Xi", "Xi is a sculptor."),
        document(14, "Omicron", "Omicron is a sculptor."),
        document(15, "Pi", "Pi is a sculptor."),
        document(16, "Rho", "Rho is a sculptor."),
        document(17, "Sigma", "Sigma is a sculptor."),
        document(18, "Tau", "Tau is a sculptor."),
        document(19, "Upsilon", "Upsilon is a sculptor."),
        document(20, "Phi", "Phi is a painter."),
        document(21, "Chi", "Chi is a sculptor."),
        document(22, "Psi", "Psi is a painter."),
        document(23, "Omega", "Omega is a sculptor."),
        document(24, "A One", "A One is a painter."),
        document(25, "B One", "B One is a sculptor."),
        document(26, "C One", "C One is a painter."),
        document(27, "D One", "D One is a sculptor."),
        document(28, "E One", "E One is a painter."),
        document(29, "F One", "F One is a sculptor."),
        document(30, "G One", "G One is a painter."),
        document(31, "H One", "H One is a sculptor."),
        document(32, "I One", "I One is a painter."),
        document(33, "J One", "J One is a sculptor."),
        document(34, "K One", "K One is a painter."),
        document(35, "L One", "L One is a sculptor."),
        document(36, "M One", "M One is a painter."),
        document(37, "N One", "N One is a sculptor."),
        document(38, "O One", "O One is a painter."),
        document(39, "P One", "P One is a sculptor."),
    )
    frequencies = Counter({("painter",): 2, ("sculptor",): 2})
    edges, diagnostics = enumerate_edges(documents, frequencies)
    assert diagnostics["witness_regeneration_failures"] == 0
    alpha_painter = [
        edge for edge in edges if edge[0] == ("alpha", ("painter",))
    ]
    assert len(alpha_painter) == 2


def test_independent_graph_constructors_agree():
    u1 = ("alpha", ("painter",))
    u2 = ("beta", ("painter",))
    p1 = ("<S>", "is", "a", "<V>")
    p2 = ("<S>", "worked", "as", "<V>")
    p3 = ("<S>", "became", "<V>")
    edges = {(u1, p1), (u1, p2), (u2, p1), (u2, p2), (u2, p3)}
    _, nodes_a, edges_a = inverted_graph(edges)
    nodes_b, edges_b = set_intersection_graph(edges)
    assert nodes_a == nodes_b
    assert edges_a == edges_b
    assert connected_components(nodes_a, edges_a) == ((p1, p2),)


def test_singleton_pattern_is_not_a_graph_node():
    u1 = ("alpha", ("painter",))
    u2 = ("beta", ("painter",))
    p1 = ("<S>", "is", "a", "<V>")
    p2 = ("<S>", "worked", "as", "<V>")
    edges = {(u1, p1), (u2, p1), (u1, p2)}
    _, nodes, graph_edges = inverted_graph(edges)
    assert nodes == {p1}
    assert graph_edges == set()


def test_polysemous_shared_tuple_merges_components_as_declared_failure():
    u1 = ("alpha", ("bank",))
    u2 = ("beta", ("bank",))
    p_finance = ("<S>", "worked", "at", "<V>")
    p_river = ("<S>", "stood", "on", "the", "<V>")
    edges = {
        (u1, p_finance),
        (u2, p_finance),
        (u1, p_river),
        (u2, p_river),
    }
    _, nodes, graph_edges = inverted_graph(edges)
    assert connected_components(nodes, graph_edges) == (tuple(sorted((p_finance, p_river))),)


def test_graph_is_invariant_to_edge_order():
    u1 = ("alpha", ("painter",))
    u2 = ("beta", ("painter",))
    p1 = ("<S>", "is", "a", "<V>")
    p2 = ("<S>", "worked", "as", "<V>")
    edge_list = [(u1, p1), (u1, p2), (u2, p1), (u2, p2)]
    _, nodes_a, edges_a = inverted_graph(edge_list)
    _, nodes_b, edges_b = inverted_graph(reversed(edge_list))
    assert nodes_a == nodes_b
    assert edges_a == edges_b
    assert connected_components(nodes_a, edges_a) == connected_components(nodes_b, edges_b)
