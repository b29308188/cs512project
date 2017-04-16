from nltk.corpus import wordnet as wn
if __name__ == "__main__":
    syn_relations = ["hyponyms", "hypernyms", "entailments", "causes", "also_sees", "attributes", 
                "member_holonyms", "member_meronyms","part_holonyms", "part_meronyms","substance_holonyms", "substance_meronyms",\
                       "instance_hypernyms", "instance_hyponyms",  "region_domains", "similar_tos", "topic_domains", "usage_domains", "verb_groups"]
    lem_relations = ["antonyms", "derivationally_related_forms", "pertainyms"]
    S = set()
    for s1 in wn.all_synsets():
        for r in syn_relations:
            for s2 in getattr(s1, r)():
                S.add((s1.name(), s2.name(), r))
        for r in lem_relations:
            for l1 in s1.lemmas():
                for l2 in getattr(l1, r)():
                    s2 = l2.synset()
                    S.add((s1.name(), s2.name(), r))
    
    for h, t, r in sorted(list(S)):
        if h != t:
            print h, t, r
