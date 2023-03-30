from random import Random

from faker import Faker

from parquetranger import ObjIngestor

rng = Random(742)

fake = Faker()

Faker.seed(742)

col_union = [
    "ascii_email",
    "bothify",
    "company_email",
    "currency_code",
    "first_name",
    "month",
    "pricetag",
    "random_digit_or_empty",
    "random_letters",
    "secondary_address",
    "state",
    "street_address",
    "street_suffix",
    "zipcode",
]

posses = [rng.choices(col_union, k=rng.randint(5, 12)) for _ in range(7)]


def get_objects(n):
    for _ in range(n):
        keys = rng.choice(posses)
        obj = {k: getattr(fake, k)() for k in keys}
        if rng.random() < 0.01:
            obj["related-list"] = list(get_objects(rng.randint(1, 10)))
        if rng.random() < 0.02:
            obj["related-dic"] = next(get_objects(1))
        if rng.random() < 0.8:
            obj["id_"] = fake.company_email()
        yield obj


def test_ingestor(tmp_path):
    with ObjIngestor(tmp_path / "data", root_id_key="id_") as ing:
        for o in get_objects(10_000):
            ing.ingest(o)

    with ObjIngestor(tmp_path / "data", root_id_key="id_", force_key=True) as ing:
        for o in get_objects(1000):
            ing.ingest(o)


def test_ingestor_add_key(tmp_path):
    with ObjIngestor(tmp_path / "data") as ing:
        ing.ingest({"id_": "abc", "X": 10})

    with ObjIngestor(tmp_path / "data", root_id_key="id_", force_key=True) as ing:
        ing.ingest({"X": 20})


def test_empty(tmp_path):
    with ObjIngestor(tmp_path / "data") as ing:
        ing.ingest({})
