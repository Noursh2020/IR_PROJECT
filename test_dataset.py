import ir_datasets

for name in [
    "beir/webis-touche2020",
]:
    ds = ir_datasets.load(name)

    print("\n", name)
    print("docs:", ds.docs_count())
    print("queries:", ds.queries_count())
    print("qrels:", ds.has_qrels())