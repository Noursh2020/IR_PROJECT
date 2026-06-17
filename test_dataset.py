import ir_datasets

for name in [
    "msmarco-passage/dev/small",
    "beir/nq"
]:
    ds = ir_datasets.load(name)

    print("\n", name)
    print("docs:", ds.has_docs())
    print("queries:", ds.has_queries())
    print("qrels:", ds.has_qrels())